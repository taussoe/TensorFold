"""Flash Next's long prefill chunks (TF_QWEN4_PREFILL_ROWS): the grouped expert kernels read an expert's weights
once for two member tiles (MT=2), and a (row, expert) pair's bits must not change."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_kernels import DEV, _experts, _moe_rows  # noqa: E402  (pytest puts tests/cuda on sys.path)
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402


@pytest.mark.parametrize("rows", [200, 700])
def test_two_member_tiles_keep_the_bits(rows):
    ex, _ = _experts(64, 640, 2560)
    torch.manual_seed(rows)
    router_rows = (torch.randn((65, 2560), device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn((rows, 2560), device=DEV).to(torch.bfloat16)
    ref = _moe_rows(x, router_rows, ex, rows)                     # rows > 128: MT=2 by default
    y2 = ref.y[:rows].clone()
    xs = qmm.group_sums(x)
    qmm.moe_gateup(x, xs, ex, ref.group, ref.act, ref.axs, mt=1)
    qmm.moe_down(ref.act, ref.axs, ex, ref.group, ref.y, mt=1)
    assert torch.equal(ref.y[:rows], y2)
    one = _moe_rows(x[rows // 2:rows // 2 + 1], router_rows, ex, 1)
    assert torch.equal(one.y[0], y2[rows // 2])
