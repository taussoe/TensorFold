"""Where a prefill's time goes: TF_GLM_PROFILE=1 times each block kind of every prefill chunk (with a device sync
around each, so only for measuring) and prints the totals after each prefill. Off by default, and never active
inside decode rounds or CUDA-graph capture."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

ENABLED = os.environ.get("TF_GLM_PROFILE", "0") == "1"
active = False                     # set only around prefill chunks
totals: dict[str, float] = {}


@contextmanager
def timed(name: str):
    if not (ENABLED and active):
        yield
        return
    torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - t


def report(tokens: int) -> None:
    if not ENABLED or not totals:
        return
    total = sum(v for k, v in totals.items() if ":" not in k)          # "dsa: x" parts are inside "dsa (total)"
    parts = ", ".join(f"{k} {v:.2f}s ({100 * v / total:.0f}%)" for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))
    print(f"[tensorfold] prefill profile, {tokens} tokens in {total:.1f}s timed ({tokens / total:.0f} tok/s): {parts}",
          flush=True)
    totals.clear()
