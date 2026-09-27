"""GLM-5.3-Flash's engine past the dense limit (2,051 tokens) on the tiny synthetic checkpoint of
test_glm_engine.py: every decoded row attends to its DSA-selected tokens through the latent cache. Drafted replies
equal serial ones, steps replayed as sparse CUDA graphs give the tokens eager steps give, and a prompt resumed
from a kept state equals a fresh prefill, with long prefill chunks."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

from test_glm_engine import _checkpoint, _generate, _TwoCopies  # noqa: E402  (pytest puts tests/cuda on sys.path)

PROMPT = 2100              # past the dense limit: the first reply token is already a sparse row
CONTEXT = 2600


@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_long")
    _checkpoint(path)
    saved = os.environ.get("TF_GLM_PREFILL_ROWS")
    os.environ["TF_GLM_PREFILL_ROWS"] = "256"            # long chunks, past the 128-row matmul block
    try:
        return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=CONTEXT)
    finally:
        if saved is None:
            os.environ.pop("TF_GLM_PREFILL_ROWS", None)
        else:
            os.environ["TF_GLM_PREFILL_ROWS"] = saved


def _prompt(seed=11, n=PROMPT):
    return list(np.random.default_rng(seed).integers(0, 1000, size=n))


def test_sparse_graphs_are_used_past_the_dense_limit(engine_long):
    g = engine_long.e.graphs
    assert g is not None and g.sparse and g.sparse_mtp, "no sparse CUDA graphs captured for a long-context engine"
    before = dict(engine_long.e.replays)
    _generate(engine_long, _prompt(seed=16), None, tokens=32)
    used = {k: engine_long.e.replays[k] - before[k] for k in before}
    assert used["sparse"] > 0 and used["sparse_mtp"] > 0, used
    assert used["main"] == 0, used                  # every step of this reply is past the dense limit


@pytest.mark.parametrize("sampling", [Sampling(99, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_long_context_drafted_equals_serial(engine_long, sampling):
    prompt = _prompt()
    serial, stats = _generate(engine_long, prompt, sampling, draft=False, tokens=32)
    assert len(serial) == 32 and stats["drafts"] is False
    for policy in (None, "2", "c3:0.35"):
        drafted, _ = _generate(engine_long, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


@pytest.mark.parametrize("sampling", [Sampling(5, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_sparse_graphs_equal_eager_steps(engine_long, sampling):
    prompt = _prompt(seed=12)
    with_graphs, _ = _generate(engine_long, prompt, sampling, tokens=32)
    graphs, engine_long.e.graphs = engine_long.e.graphs, None
    try:
        eager, _ = _generate(engine_long, prompt, sampling, tokens=32)
    finally:
        engine_long.e.graphs = graphs
    assert with_graphs == eager


def test_long_prompt_resumes_like_a_fresh_prefill(engine_long):
    sampling = Sampling(21, 1.0, 20, 0.95)
    first = _prompt(seed=13, n=2080)                    # crosses the dense limit inside a 256-row chunk
    reply, _ = _generate(engine_long, first, sampling, tokens=16)
    follow = first + reply + _prompt(seed=14, n=40)
    warm, stats = _generate(engine_long, follow, sampling, tokens=16)
    assert stats["cached"] >= len(first) + len(reply) - 1
    _generate(engine_long, _prompt(seed=15, n=12), sampling, tokens=4)   # a fresh prefill: kept states go
    cold, stats = _generate(engine_long, follow, sampling, tokens=16)
    assert stats["cached"] == 0 and warm == cold
