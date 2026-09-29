"""Time disk.Store.put on GLM-5.3-Flash's real state sizes (one rank): where a prompt's write goes."""

import cProfile
import pstats
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tensorfold.families.glm5_next.cuda import disk
from tensorfold.families.glm5_next.cuda.decode import Snapshot

cap, layers, dsa = 65536, 34, 12
dev = "cuda"
st = SimpleNamespace(
    kc=[torch.zeros(cap, 512, dtype=torch.bfloat16, device=dev) for _ in range(dsa - 1)], vc=[None] * (dsa - 1),
    index=[(torch.zeros(cap, 128, dtype=torch.bfloat16, device=dev), torch.zeros(cap, 32, dtype=torch.bfloat16, device=dev),
            torch.zeros(cap // 4 + 2, 128, dtype=torch.bfloat16, device=dev)) for _ in range(dsa)],
    mtp_kc=torch.zeros(cap, 512, dtype=torch.bfloat16, device=dev), mtp_vc=None)
e = SimpleNamespace(st=st)
rec = torch.randn(layers, 32, 128, 128, device=dev)
conv = torch.randn(layers, 3, 3 * 32 * 128, device=dev).bfloat16()
store = disk.Store(Path(tempfile.mkdtemp()), 64 * 2 ** 30, "x")
ids = list(np.random.default_rng(0).integers(0, 1000, 20000))
for step in range(4):
    n = 18000 + step * 300
    snap = Snapshot(ids[:n], rec, conv, torch.zeros(1, 4096, dtype=torch.bfloat16, device=dev), n - 1, -1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    if step == 3:
        pr = cProfile.Profile()
        pr.enable()
    store.put(e, snap)
    if step == 3:
        pr.disable()
    print(f"put {n} tokens: {time.perf_counter() - t:.3f}s")
pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
