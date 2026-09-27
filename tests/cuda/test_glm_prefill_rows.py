"""Long prefill chunks (TF_GLM_PREFILL_ROWS) must not change a row's bits: every GLM kernel on the prefill path
gives a row the same result whatever the chunk size and tile settings."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def _experts(E, D, NI, gen):
    from tensorfold.families.glm5_next.cuda import qmm

    def q4(n, k):
        w = torch.randint(0, 2 ** 31 - 1, (E, n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
        s = (torch.rand((E, n, k // 64), generator=gen) * 0.02).to(torch.bfloat16)
        b = (torch.randn((E, n, k // 64), generator=gen) * 0.01).to(torch.bfloat16)
        return w.cuda(), s.cuda(), b.cuda()

    return qmm.make_experts(q4(NI, D), q4(NI, D), q4(D, NI))


@cuda
@pytest.mark.parametrize("R", [16, 200, 1024])
def test_moe_member_tile_does_not_change_bits(R):
    from tensorfold.families.glm5_next.cuda import glue, qmm

    gen = torch.Generator().manual_seed(R)
    E, D, NI, K = 16, 512, 256, 4
    ex = _experts(E + 1, D, NI, gen)          # the last one is the shared expert (slot K)
    x = torch.randn((R, D), generator=gen).to(torch.bfloat16).cuda()
    xs = qmm.group_sums(x)
    logits = torch.randn((R, E), generator=gen).cuda()
    pick = torch.empty((R, K + 1), dtype=torch.int32, device="cuda")
    wts = torch.empty((R, K + 1), dtype=torch.float32, device="cuda")
    grp = qmm.Group(torch.zeros((E + 1,), dtype=torch.int32, device="cuda"),
                    torch.zeros((1,), dtype=torch.int32, device="cuda"),
                    torch.full((E + 1, R), -1, dtype=torch.int32, device="cuda"))
    glue.select(logits, torch.zeros((E,), device="cuda"), pick, wts, grp.ids, grp.count, grp.members, K, E, 1.0, True)
    outs = {}
    for bm, mt in ((16, 1), (32, 1), (32, 2), (16, 2)):
        act = torch.zeros((R, K + 1, NI), dtype=torch.bfloat16, device="cuda")
        axs = torch.zeros((R, K + 1, NI // 64), dtype=torch.float32, device="cuda")
        y = torch.zeros((R, K + 1, D), dtype=torch.float32, device="cuda")
        qmm.moe_gateup(x, xs, ex, grp, act, axs, 7.0, bm=bm, mt=mt)
        qmm.moe_down(act, axs, ex, grp, y, bm=bm, mt=mt)
        outs[(bm, mt)] = y
    for key in outs:
        assert torch.equal(outs[(16, 1)], outs[key]), f"member tile {key[0]} x {key[1]} changed bits"


@cuda
@pytest.mark.parametrize("m", [64, 129, 1024])
def test_dense_matmul_rows_past_128_keep_bits(m):
    """A row's result in an m-row call equals its result alone."""
    from tensorfold.families.glm5_next.cuda import qmm

    gen = torch.Generator().manual_seed(m)
    n, k = 1536, 4096
    w = torch.randint(0, 2 ** 31 - 1, (n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
    s = (torch.rand((n, k // 64), generator=gen) * 0.02).to(torch.bfloat16)
    b = (torch.randn((n, k // 64), generator=gen) * 0.01).to(torch.bfloat16)
    q = qmm.make_q4(w.cuda(), s.cuda(), b.cuda())
    x = torch.randn((m, k), generator=gen).to(torch.bfloat16).cuda()
    full = qmm.matmul(x, q)
    for r in (0, m // 2, m - 1):
        one = qmm.matmul(x[r:r + 1].contiguous(), q)
        assert torch.equal(full[r], one[0]), f"row {r} of {m}"
