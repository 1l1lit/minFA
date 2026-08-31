import sys
import torch
import torch_npu
torch_npu.npu.set_device(5)
from flash_attn_npu import flash_attn_func


def run_fwd(q, k, v):
    """
    标准 Flash Attention 前向推理
    输入形状: (batch_size, seqlen, nheads, headdim)
    """
    output = flash_attn_func(
        q=q,
        k=k,
        v=v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False
    )
    return output


def main():
    batch_size, seqlen, nheads, nheadskv, headdim = 320, 150, 24, 4, 128

    # 随机初始化 Q, K, V
    q = torch.randn(batch_size, seqlen, nheads, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    k = torch.randn(batch_size, seqlen, nheadskv, headdim, device='npu', dtype=torch.float16, requires_grad=False)
    v = torch.randn(batch_size, seqlen, nheadskv, headdim, device='npu', dtype=torch.float16, requires_grad=False)

    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)

    for _ in range(10): # warm-up runs
        run_fwd(q, k, v)

    torch.npu.synchronize()
    start_event.record()

    for _ in range(100): # timed runs
        run_fwd(q, k, v)

    end_event.record()
    end_event.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average time for 100 runs: {elapsed_time_ms / 100:.3f} ms")
    return 0


if __name__ == '__main__':
    sys.exit(main())