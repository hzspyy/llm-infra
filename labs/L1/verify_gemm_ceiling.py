#!/usr/bin/env python3
"""L1.2 lab · 把「TFLOPS」换算成时钟无关的硬件常数，才能判断一个数字可不可信。

TFLOPS 会随时钟浮动，而时钟会因为功耗墙、温度、负载类型而变。
所以跨实验比较 TFLOPS 是没有意义的。真正该比的是：

    FLOP / clock / SM

这是一个**硬件常数**，与频率无关。NVIDIA 的标称值就是「这个常数 × 基频 × SM 数」。
本脚本同时采样时钟并做正确性校验，把 cuBLAS 的数字放到同一个尺子上。
"""
import statistics
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sm_clock_ghz():
        return pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_SM) / 1000.0

    def power_w():
        return pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0
except Exception:                                   # noqa: BLE001
    def sm_clock_ghz():
        return float("nan")

    def power_w():
        return float("nan")


def bench(fn, flop, sms, iters=30, warm=10):
    """时钟必须在 kernel **运行期间**采样。

    第一版在 torch.cuda.synchronize() 之后才读 NVML，
    那时 GPU 已经掉回空闲频率，读出 0.18 GHz / 27 W —— 换算出 7260 FLOP/clk/SM
    这种荒谬数字。GEMM 只跑几毫秒，采样必须和它并发。
    """
    import threading
    import time as _t

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()

    stop = threading.Event()
    clks, pws = [], []

    def sampler():
        while not stop.is_set():
            clks.append(sm_clock_ghz())
            pws.append(power_w())
            _t.sleep(0.002)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    times = []
    try:
        # 连续跑满一段时间，让频率进入稳态，同时收集逐次耗时
        for _ in range(iters):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(20):            # 一组 20 次，中间不同步
                fn()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) / 20)
    finally:
        stop.set()
        th.join(timeout=1)

    ms = statistics.median(times)
    # 丢掉前 30% 的样本（频率爬升期），取稳态中位数
    tail = clks[len(clks) // 3:] or clks
    ghz = statistics.median(tail) if tail else float("nan")
    watt = statistics.median(pws[len(pws) // 3:] or pws) if pws else float("nan")
    tflops = flop / (ms * 1e-3) / 1e12
    return {"ms": ms, "tflops": tflops, "ghz": ghz, "watt": watt,
            "flop_per_clk_per_sm": tflops * 1e12 / (sms * ghz * 1e9)}


def main():
    p = torch.cuda.get_device_properties(0)
    sms = p.multi_processor_count
    s = 8192
    flop = 2 * s ** 3
    print(f"=== {p.name}  SM={sms}   矩阵 {s}x{s}x{s}, FLOP={flop/1e12:.3f}T\n")

    af = torch.randn(s, s, device="cuda") / 8
    bf = torch.randn(s, s, device="cuda") / 8
    abf, bbf = af.to(torch.bfloat16), bf.to(torch.bfloat16)
    a8 = af.to(torch.float8_e4m3fn)
    b8 = bf.to(torch.float8_e4m3fn).t().contiguous().t()
    sc = torch.tensor(1.0, device="cuda")

    # 正确性：确认 fp8 路径真的在算这个矩阵乘，而不是被跳过/退化
    ref = (abf @ bbf).float()
    got = torch._scaled_mm(a8, b8, scale_a=sc, scale_b=sc,
                           out_dtype=torch.bfloat16).float()
    rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
    verdict = "确实算了" if rel < 0.3 else "可疑"
    print(f"[正确性] fp8 结果 vs bf16 参考：相对误差 {rel:.4f}  -> {verdict}")
    print(f"          bf16 均值 {ref.abs().mean():.4f} / fp8 均值 {got.abs().mean():.4f}\n")

    print(f"{'路径':18s} {'ms':>8s} {'TFLOPS':>9s} {'GHz':>6s} {'W':>6s} {'FLOP/clk/SM':>12s}")
    for name, fn in [
        ("bf16 torch.mm", lambda: torch.mm(abf, bbf)),
        ("fp8 _scaled_mm", lambda: torch._scaled_mm(a8, b8, scale_a=sc, scale_b=sc,
                                                    out_dtype=torch.bfloat16)),
    ]:
        r = bench(fn, flop, sms)
        print(f"{name:18s} {r['ms']:8.3f} {r['tflops']:9.1f} {r['ghz']:6.2f} "
              f"{r['watt']:6.0f} {r['flop_per_clk_per_sm']:12.0f}")

    print("\n参照（labs/L1/tensor_core.cu 的纯 mma.sync 发射，不碰内存）：")
    print("    bf16 f32累加   511 FLOP/clk/SM")
    print("    fp8  f32累加  1022 FLOP/clk/SM")
    print("这两个数就是 NVIDIA 标称值 ÷ (SM数 × 基频)，是硬件常数。")
    print("cuBLAS 若超过它，一定是测法有问题（或用了别的指令/稀疏）。")


if __name__ == "__main__":
    main()
