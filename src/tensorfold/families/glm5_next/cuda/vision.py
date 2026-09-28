"""GLM-5.3-Flash's image input: the checkpoint processor's canvas and patches, and its vision tower (bf16, rank 0).

A prompt carries an image as a run of placeholder tokens; the tower's rows replace their embeddings. In the engine
those tokens are keyed by the image's hash (``key``: a negative id), so a cached conversation resumes only when its
images are the same ones; staging maps keyed ids back to the placeholder token.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PATCH, TEMPORAL, MERGE = 14, 2, 2
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
MIN_TOKENS, MAX_TOKENS = 16, 8000                 # processor_config.json's image token budget
VISION_FILE = "vision.safetensors"                # the tower beside a rank folder's weights (``split.py``)
PREFIX = "model.visual."


@dataclass(slots=True)
class Image:
    """One prepared image: its patches [t*h*w, 3*2*14*14] fp32, grid (t, h, w) and content hash."""

    patches: torch.Tensor
    grid: tuple[int, int, int]
    digest: bytes

    @property
    def tokens(self) -> int:
        t, h, w = self.grid
        return t * h * w // (MERGE * MERGE)

    @property
    def key(self) -> int:
        """The negative id its placeholder tokens carry in the engine (31 bits of the hash)."""

        return -(1 + int.from_bytes(self.digest[:4], "little") % (2 ** 31 - 2))


# -- the processor (Glm5NextImageProcessor, resize_mode "pad") ---------------------------------------------------
def _ceil(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _fit(t: int, h: int, w: int, factor: int, max_pixels: int) -> tuple[int, int]:
    """The largest proportional canvas whose upward-aligned size fits the budget (binary search on the height)."""

    low, high = 1, h
    best = factor, factor
    while low <= high:
        ch = (low + high) // 2
        cw = max(1, math.floor(w * ch / h))
        ah, aw = _ceil(ch, factor), _ceil(cw, factor)
        if t * ah * aw <= max_pixels:
            best = ah, aw
            low = ch + 1
        else:
            high = ch - 1
    return best


def canvas(h: int, w: int, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS) -> tuple[int, int]:
    """The processor's ``smart_resize``: the height and width rounded up to 28, refit into the token budget."""

    factor = PATCH * MERGE
    per = TEMPORAL * factor * factor
    min_pixels, max_pixels = min_tokens * per, max_tokens * per
    t = TEMPORAL
    hb, wb = _ceil(h, factor), _ceil(w, factor)
    if t * hb * wb > max_pixels:
        return _fit(t, h, w, factor, max_pixels)
    if t * hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        hb, wb = _ceil(max(1, math.ceil(h * beta)), factor), _ceil(max(1, math.ceil(w * beta)), factor)
        if t * hb * wb > max_pixels:
            return _fit(t, h, w, factor, max_pixels)
    return hb, wb


def decode(data: bytes):
    """Encoded bytes -> ``ImageInput`` (RGB, transparency over white), as the server decodes an image part."""

    from PIL import Image as PILImage

    from tensorfold.vision.images import ImageInput

    try:
        img = PILImage.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 - any undecodable upload is the caller's error
        raise ValueError(f"the image could not be decoded: {exc}") from None
    if img.mode != "RGB":                         # transformers' convert_to_rgb: transparency over white
        rgba = img.convert("RGBA")
        white = PILImage.new("RGBA", rgba.size, (255, 255, 255))
        img = PILImage.alpha_composite(white, rgba).convert("RGB")
    return ImageInput(img.width, img.height, img.tobytes())


def prepare(image, *, max_tokens: int = MAX_TOKENS) -> Image:
    """Fit an image onto the canvas keeping the aspect ratio (bicubic, zero padding right and bottom), normalize, patch.

    ``image``: an ``ImageInput`` (decoded RGB, as the server's ``load_images`` gives it) or encoded bytes.
    """

    from tensorfold.vision.images import ImageInput

    if not isinstance(image, ImageInput):
        image = decode(image)
    digest = bytes.fromhex(image.content_hash)
    x = torch.frombuffer(bytearray(image.pixels), dtype=torch.uint8).view(image.height, image.width, 3)
    x = x.permute(2, 0, 1)                                             # [3, H, W] uint8
    H, W = x.shape[1:]
    ch, cw = canvas(H, W, max_tokens=max_tokens)
    scale = min(ch / H, cw / W)
    if TEMPORAL * H * W >= MIN_TOKENS * TEMPORAL * (PATCH * MERGE) ** 2:
        scale = min(1.0, scale)                   # small images are enlarged only below the minimum budget
    th, tw = max(1, min(ch, math.floor(H * scale))), max(1, min(cw, math.floor(W * scale)))
    if (th, tw) != (H, W):                        # torchvision's tensor resize: float bicubic, antialiased, rounded
        x = F.interpolate(x[None].float(), size=(th, tw), mode="bicubic", align_corners=False, antialias=True)[0]
        x = x.round_().clamp_(0, 255).to(torch.uint8)
    x = F.pad(x, (0, cw - tw, 0, ch - th), value=0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    x = (x.float() * (1 / 255.0) - mean) / std
    gh, gw = ch // PATCH, cw // PATCH
    frames = x[None].expand(TEMPORAL, 3, ch, cw)                   # a still image repeats its frame
    p = frames.reshape(1, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)                       # (t, gh, gw, mh, mw, C, tp, ph, pw)
    return Image(p.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH).contiguous(), (1, gh, gw), digest)


@dataclass(slots=True)
class Prepared:
    """A prompt with its images: ``token_ids`` carry each image as its keyed run, ``images`` in prompt order."""

    token_ids: list[int]
    images: list[Image]


class Frontend:
    """The server's image frontend (``App.vision``): the rendered prompt's placeholders become keyed image runs."""

    allow_urls = False

    def __init__(self, tokenizer, image_token: int) -> None:
        self.tok, self.image_token = tokenizer, image_token

    def prepare(self, text: str, images: list, max_prompt_tokens: int | None = None) -> Prepared:
        prepared = [prepare(image) for image in images]
        prompt, k = [], 0
        for t in self.tok.encode(text, add_special_tokens=False).ids:
            if t == self.image_token and k < len(prepared):
                prompt.extend([prepared[k].key] * prepared[k].tokens)
                k += 1
            else:
                prompt.append(t)
        if k != len(prepared) or self.image_token in prompt:
            raise ValueError(f"the chat template placed {k + prompt.count(self.image_token)} images "
                             f"for {len(prepared)} image parts")
        if max_prompt_tokens is not None and len(prompt) > max_prompt_tokens:
            raise ValueError(f"the prompt with its images is {len(prompt)} tokens, over the "
                             f"{max_prompt_tokens}-token context")
        return Prepared(prompt, prepared)

    def continued(self, prepared: Prepared, ids: list[int]) -> Prepared:
        """A continuation of the prompt (a tool-call gate) keeps its images: their runs sit in the same place."""

        return Prepared(list(ids), prepared.images)


# -- the vision tower ---------------------------------------------------------------------------------------------
def _tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """The ``model.visual.*`` tensors: a rank folder's vision file, else the checkpoint's own files."""

    from safetensors import safe_open

    path = model_dir / VISION_FILE
    if path.exists():
        files = {path: None}
    else:
        index = model_dir / "model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(f"{model_dir}: no {VISION_FILE} and no checkpoint index to find the vision tower")
        names = json.loads(index.read_text())["weight_map"]
        files = {model_dir / f: None for n, f in names.items() if n.startswith(PREFIX)}
        if not files:
            raise FileNotFoundError(f"{model_dir}: this checkpoint has no vision tower")
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for n in h.keys():
                if n.startswith(PREFIX):
                    out[n[len(PREFIX):]] = h.get_tensor(n)
    return out


def available(model_dir: Path) -> bool:
    if (model_dir / VISION_FILE).exists():
        return True
    index = model_dir / "model.safetensors.index.json"
    return index.exists() and any(n.startswith(PREFIX) for n in json.loads(index.read_text())["weight_map"])


def tower_bytes(model_dir: Path) -> int:
    """The tower's weights as loaded (bf16), for the startup memory estimate."""

    from .split import read_header

    path = model_dir / VISION_FILE
    if path.exists():
        header = read_header(path)[0]
    else:
        index = model_dir / "model.safetensors.index.json"
        if not index.exists():
            return 0
        names = json.loads(index.read_text())["weight_map"]
        header = {}
        for f in sorted({f for n, f in names.items() if n.startswith(PREFIX)}):
            header.update(read_header(model_dir / f)[0])
    return sum(math.prod(v["shape"]) * 2 for k, v in header.items() if k.startswith(PREFIX))


# rows of the tower's MLP, of its merger and queries of its attention at a time: the scratch of an 8,000-token
# image (32,000 patches, fp32) peaks at 1.18 GiB, in its attention
ROWS = 4096
MERGE_ROWS = 2048
QUERIES = 4096
WORKSPACE = 5 * 2 ** 28                           # 1.25 GiB for the startup estimate


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _tf32(x: torch.Tensor) -> torch.Tensor:
    """x's leading 11 significant bits (TF32 keeps them exactly); x - _tf32(x) holds the rest."""

    return (x.contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _attention(device: torch.device):
    """The memory-efficient attention kernel on CUDA (fp32 products at fp32 accuracy); any kernel elsewhere."""

    from contextlib import nullcontext

    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])


class Tower:
    """The vision transformer (24 blocks, 2D rotary, q/k norms), the 2x2 downsample and the SwiGLU merger.

    Its weights stay bf16 as stored; every product is fp32. The tower amplifies rounding: run all in bf16 (as
    transformers and vLLM run it) its rows for a photo are 7% off the exact ones (some below cosine 0.84), with
    TF32 products 1.7%. NVIDIA's containers force TF32 on cuBLAS, so a product is two TF32 products (an input's
    leading bits, then the rest; bf16 weights are exact in TF32), and attention runs the memory-efficient
    kernel, whose fp32 products are split alike.
    """

    def __init__(self, model_dir: Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        v = cfg["vision_config"]
        self.depth, self.dim, self.heads = int(v["depth"]), int(v["hidden_size"]), int(v["num_heads"])
        self.out = int(v["out_hidden_size"])
        self.eps = float(v.get("rms_norm_eps", 1e-5))
        self.limit = float(v.get("swiglu_limit") or 10.0)
        if (int(v["patch_size"]), int(v["temporal_patch_size"]), int(v["spatial_merge_size"])) != (PATCH, TEMPORAL,
                                                                                                    MERGE):
            raise ValueError("this vision tower's patch geometry differs from GLM-5.3-Flash's")
        # matrices in ``dtype``; norms and biases fp32
        t = {k: x.to(device=device, dtype=dtype if x.dim() > 1 else torch.float32)
             for k, x in _tensors(Path(model_dir)).items()}
        t["patch_embed.proj.weight"] = t["patch_embed.proj.weight"].reshape(self.dim, -1)
        t["downsample.weight"] = t["downsample.weight"].reshape(self.out, -1)
        self.t = t
        hd = self.dim // self.heads
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd // 2, 2, dtype=torch.float32, device=device)
                                           / (hd // 2)))
        self.device = device

    def _lin(self, x: torch.Tensor, name: str, bias: bool = True, part: slice = slice(None)) -> torch.Tensor:
        """x @ W[part].T (+ b[part]), fp32."""

        w = self.t[name + ".weight"][part].float()
        hi = _tf32(x)
        y = F.linear(hi, w)
        y += F.linear(torch.sub(x, hi, out=hi), w)
        return y.add_(self.t[name + ".bias"][part]) if bias else y

    def _rotary(self, grid: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        t, h, w = grid
        hp = torch.arange(h, device=self.device)[:, None].expand(h, w)
        wp = torch.arange(w, device=self.device)[None, :].expand(h, w)

        def order(p):
            return p.reshape(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).reshape(-1)

        pos = torch.stack([order(hp), order(wp)], dim=-1).repeat(t, 1).float()    # [N, 2]
        freqs = (pos[:, :, None] * self.inv_freq[None, None, :]).reshape(pos.shape[0], -1)   # [N, hd/2]
        return freqs.cos(), freqs.sin()

    def _block(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        t, p = self.t, f"blocks.{i}."
        N, H = x.shape[0], self.heads
        hd = self.dim // H
        h = _rms(x, t[p + "norm1.weight"], self.eps)
        c, s = cos[:, None, :], sin[:, None, :]

        def rope(z):                               # neox halves over the whole head: [h freqs | w freqs] twice
            a, b = z[..., :hd // 2], z[..., hd // 2:]
            return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

        def head(i, norm=None):                    # q, k or v: [1, H, N, hd] (one at a time: less scratch)
            z = self._lin(h, p + "attn.qkv", part=slice(i * self.dim, (i + 1) * self.dim)).view(N, H, hd)
            if norm is not None:
                z = rope(_rms(z, t[p + norm], 1e-5))
            return z.transpose(0, 1).contiguous()[None]

        q, k, v = head(0, "attn.q_norm.weight"), head(1, "attn.k_norm.weight"), head(2)
        del h
        att = torch.empty_like(q)
        with _attention(q.device):
            for a in range(0, N, QUERIES):
                att[:, :, a:a + QUERIES] = F.scaled_dot_product_attention(q[:, :, a:a + QUERIES], k, v)
        x = x + self._lin(att[0].transpose(0, 1).reshape(N, self.dim), p + "attn.proj")
        for a in range(0, N, ROWS):
            h = _rms(x[a:a + ROWS], t[p + "norm2.weight"], self.eps)
            act = _swiglu(self._lin(h, p + "mlp.gate_proj"), self._lin(h, p + "mlp.up_proj"), self.limit)
            x[a:a + ROWS] += self._lin(act, p + "mlp.down_proj")
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        t = self.t
        x = self._lin(x, "merger.proj", bias=False)
        x = F.gelu(F.layer_norm(x, (self.out,), t["merger.post_projection_norm.weight"],
                                t["merger.post_projection_norm.bias"], 1e-5))
        act = _swiglu(self._lin(x, "merger.gate_proj", False), self._lin(x, "merger.up_proj", False), self.limit)
        return self._lin(act, "merger.down_proj", bias=False)

    @torch.no_grad()
    def encode(self, image: Image) -> torch.Tensor:
        """An image's rows for the language model: [tokens, out_hidden] bf16."""

        try:
            return self._encode(image)
        finally:
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()            # the scratch back for the prefill

    def _encode(self, image: Image) -> torch.Tensor:
        x = self._lin(image.patches.to(self.device), "patch_embed.proj")
        cos, sin = self._rotary(image.grid)
        for i in range(self.depth):
            x = self._block(i, x, cos, sin)
        x = _rms(x, self.t["post_layernorm.weight"], self.eps)
        m = x.shape[0] // (MERGE * MERGE)
        x = x.view(m, MERGE, MERGE, self.dim).permute(0, 3, 1, 2).reshape(m, -1)    # (C, kh, kw) like the conv
        x = self._lin(x, "downsample")
        return torch.cat([self._merge(x[a:a + MERGE_ROWS]).to(torch.bfloat16) for a in range(0, m, MERGE_ROWS)])


def write_vision(src: Path, out: Path) -> int:
    """Copy the checkpoint's ``model.visual.*`` tensors into OUT/vision.safetensors (a rank 0 folder); their count."""

    from .split import write, read_header

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = []
    for f in sorted({f for n, f in index.items() if n.startswith(PREFIX)}):
        header, base = read_header(src / f)
        mm = np.memmap(src / f, dtype=np.uint8, mode="r")
        for name in sorted(header):
            if name.startswith(PREFIX):
                a, b = header[name]["data_offsets"]
                tensors.append((name, header[name]["dtype"], header[name]["shape"], np.array(mm[base + a:base + b])))
    if tensors:
        write(str(out / VISION_FILE), tensors, None)
    return len(tensors)

