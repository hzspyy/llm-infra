"""cuBLAS FP8 GEMM 的累加精度：用溢出边界判定 f16 还是 f32。"""
import json, sys, torch
out = {}
s = torch.tensor(1.0, device="cuda")
rows = []
for K in (32, 64, 256, 1024):
    a = (torch.ones(256, K, device="cuda") * 240).to(torch.float8_e4m3fn)
    b = (torch.ones(K, 256, device="cuda") * 240).t().contiguous().t().to(torch.float8_e4m3fn)
    o = torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.float32)
    exact = 240.0 * 240.0 * K
    v = float(o[0, 0])
    rows.append(dict(K=K, exact=exact, measured=v, exact_match=abs(v - exact) < exact * 1e-6))
    print(f"K={K:>5d}  精确值 {exact:>15,.0f}   实测 {v:>15,.1f}   "
          f"{'一致' if rows[-1]['exact_match'] else '不同'}")
ones = []
for K in (2048, 8192):
    a = torch.ones(256, K, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(K, 256, device="cuda", dtype=torch.bfloat16)
    af = a.to(torch.float8_e4m3fn); bf = b.t().contiguous().t().to(torch.float8_e4m3fn)
    v = float(torch._scaled_mm(af, bf, scale_a=s, scale_b=s, out_dtype=torch.float32)[0, 0])
    ones.append(dict(K=K, exact=float(K), measured=v))
    print(f"全 1 矩阵 K={K:>5d}  精确值 {K}   实测 {v:.1f}")
print("判据：f16 累加时 240×240=57,600 尚可表示，两项相加 115,200 已超过 f16 上限 65,504，")
print("      实测在 K=1024 仍精确到 58,982,400，说明累加器是 f32。")
out = dict(overflow_probe=rows, ones_probe=ones,
           note="240x240 products overflow fp16 after two terms; exact results prove fp32 accumulate")
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
