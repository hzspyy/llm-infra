#!/usr/bin/env python3
"""SmolLM2-360M 上的完整训练步：阶段时间线、重算对照、精度诊断与编译代价。

三个实验：
  timeline   真实 DataLoader → H2D → 前向 → 反向 → clip → step 的逐阶段记录，
             含 CPU 提交与 GPU 执行两套时间、显存分解、三种吞吐分母，
             并对"每层重算开关"做等价性与代价对照
  precision  FP32 / BF16 autocast / FP16+scaler 各两次尝试，定位首个非有限张量
  compile    eager 与 torch.compile 的首次编译代价、形状变化引起的重编译计数

供数用预分词的 uint16 mmap（--workdir 指向学习盘），与真实预训练数据的布局一致。
不写 checkpoint；保存与恢复的实测在 7.4。

Usage（crater，envs/serve）:
    python labs/L7/training_step_timeline.py --experiment timeline \\
        --workdir "$RUN_DIR/data" --outdir "$RUN_DIR/timeline"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE = -100
MODEL = "HuggingFaceTB/SmolLM2-360M"
SEQ, N_SAMPLES = 512, 64
MICRO_BS, ACCUM = 2, 2
LR, WD = 1e-4, 0.01


# ------------------------------------------------------------------ 供数
class TokenShardDataset(Dataset):
    """定长预分词 mmap：每条样本是 SEQ 个 uint16 token，O(1) 定位。"""

    def __init__(self, path: Path, seq: int):
        self.path, self.seq = path, seq
        self.array = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, seq)

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.array[i], dtype=np.int64))


def build_shard(workdir: Path, tokenizer, seq: int, n: int) -> Path:
    """用 tokenizer 把一段真实文本编成定长 token 流，写成 uint16 mmap。"""
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"tokens_{n}x{seq}.u16"
    if path.exists():
        return path
    base = ("Training infrastructure turns a recipe into an update. "
            "The data path, the compute path and the optimizer clock are three "
            "different systems that only meet once per step. ")
    ids: list[int] = []
    while len(ids) < n * seq:
        ids.extend(tokenizer(base * 4, add_special_tokens=False)["input_ids"])
    arr = np.asarray(ids[: n * seq], dtype=np.uint16)
    arr.tofile(path)
    return path


def make_batch(ids: torch.Tensor):
    labels = ids.clone()
    labels[:, : SEQ // 8] = IGNORE          # 前 1/8 当作 prompt，不计 loss
    return labels


def valid_targets(labels: torch.Tensor) -> int:
    shifted = F.pad(labels, (0, 1), value=IGNORE)[..., 1:]
    return int((shifted != IGNORE).sum())


# ------------------------------------------------------------------ 工具
class Phases:
    """同时记录 CPU 提交时间与 GPU 执行时间；两者不相加。"""

    def __init__(self, device):
        self.device = device
        self.cpu: list[tuple[str, float]] = []
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def mark(self, name: str):
        return _Phase(self, name)

    def gpu_ms(self) -> dict:
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for name, start, end in self.events:
            out[name] = out.get(name, 0.0) + start.elapsed_time(end)
        return {k: round(v, 3) for k, v in out.items()}

    def cpu_ms(self) -> dict:
        out: dict[str, float] = {}
        for name, dur in self.cpu:
            out[name] = out.get(name, 0.0) + dur * 1e3
        return {k: round(v, 3) for k, v in out.items()}


class _Phase:
    def __init__(self, owner: Phases, name: str):
        self.owner, self.name = owner, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        self.owner.cpu.append((self.name, time.perf_counter() - self.t0))
        self.owner.events.append((self.name, self.start, self.end))
        return False


class SavedBytes:
    """统计一次前向为反向保存了多少字节，按 dtype 分。"""

    def __init__(self):
        self.bytes: dict[str, int] = {}
        self.count = 0
        self._seen: set[int] = set()

    def __enter__(self):
        def pack(t):
            key = t.data_ptr()
            if key not in self._seen:
                self._seen.add(key)
                self.count += 1
                k = str(t.dtype).replace("torch.", "")
                self.bytes[k] = self.bytes.get(k, 0) + t.numel() * t.element_size()
            return t

        self._h = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._h.__enter__()
        return self

    def __exit__(self, *exc):
        return self._h.__exit__(*exc)

    def total(self) -> int:
        return sum(self.bytes.values())


def mem_snapshot() -> dict:
    return {"allocated_MiB": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
            "reserved_MiB": round(torch.cuda.memory_reserved() / 2 ** 20, 1),
            "max_allocated_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)}


def load_model(dtype=torch.float32, checkpointing=False):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to("cuda")
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model


# ------------------------------------------------------------------ timeline
def run_timeline(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    shard = build_shard(Path(args.workdir), tokenizer, SEQ, N_SAMPLES)
    dataset = TokenShardDataset(shard, SEQ)
    print(f"数据分片 {shard}（{len(dataset)} 条 × {SEQ} token，"
          f"{shard.stat().st_size / 2 ** 20:.1f} MiB）")

    modes = []
    for checkpointing in (False, True):
        torch.manual_seed(0)
        loader = DataLoader(dataset, batch_size=MICRO_BS, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            prefetch_factor=2 if args.workers else None,
                            persistent_workers=bool(args.workers))
        model = load_model(checkpointing=checkpointing)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                                fused=args.fused)

        it = iter(loader)
        # 预热：让 allocator、cudnn 与 worker 进入稳态，再记录被分析的那一步
        for _ in range(args.warmup):
            ids = next(it).to("cuda", non_blocking=True)
            loss = model(input_ids=ids, labels=make_batch(ids)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # 模块 forward hook 在非重入 checkpoint 的重算中不会触发，
        # 因此改为统计激活函数的实际调用次数来观察重算。
        real_silu, silu_calls = F.silu, {"n": 0}

        def counting_silu(*a, **kw):
            silu_calls["n"] += 1
            return real_silu(*a, **kw)

        F.silu = counting_silu

        ph = Phases("cuda")
        mem = {"步开始": mem_snapshot()}
        saved = SavedBytes()
        samples = tokens = 0
        losses = []
        for micro in range(ACCUM):
            with ph.mark("fetch"):
                host = next(it)
            with ph.mark("h2d"):
                ids = host.to("cuda", non_blocking=True)
            labels = make_batch(ids)
            tokens += valid_targets(labels)
            samples += ids.shape[0]
            with ph.mark("forward"):
                if micro == 0:
                    with saved:
                        out = model(input_ids=ids, labels=labels)
                else:
                    out = model(input_ids=ids, labels=labels)
                loss = out.loss / ACCUM
            if micro == 0:
                mem["前向后"] = mem_snapshot()
            losses.append(float(out.loss.detach()))
            with ph.mark("backward"):
                loss.backward()
            if micro == 0:
                mem["首次反向后"] = mem_snapshot()
        with ph.mark("clip"):
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        with ph.mark("optimizer"):
            opt.step()
        mem["optimizer 后"] = mem_snapshot()
        with ph.mark("zero_grad"):
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        mem["zero_grad 后"] = mem_snapshot()
        F.silu = real_silu

        gpu, cpu = ph.gpu_ms(), ph.cpu_ms()
        wall = sum(cpu.values())
        opt_bytes = sum(t.numel() * t.element_size()
                        for st in opt.state.values() for t in st.values()
                        if torch.is_tensor(t))
        modes.append({
            "gradient_checkpointing": checkpointing,
            "silu_calls": silu_calls["n"],
            "loss_per_microbatch": [round(v, 6) for v in losses],
            "grad_norm": round(gnorm, 6),
            "gpu_ms": gpu, "cpu_ms": cpu,
            "gpu_total_ms": round(sum(gpu.values()), 3),
            "cpu_total_ms": round(wall, 3),
            "saved_forward_bytes": saved.bytes,
            "saved_forward_MiB": round(saved.total() / 2 ** 20, 2),
            "saved_tensor_count": saved.count,
            "memory": mem,
            "optimizer_state_MiB": round(opt_bytes / 2 ** 20, 1),
            "throughput": {
                "samples": samples, "effective_tokens": tokens,
                "optimizer_updates": 1,
                "samples_per_s": round(samples / (wall / 1e3), 2),
                "effective_tokens_per_s": round(tokens / (wall / 1e3), 1),
                "updates_per_s": round(1 / (wall / 1e3), 3),
            },
        })
        del model, opt, loader, it
        torch.cuda.empty_cache()

    print(f"\n{'配置':<16}{'silu 调用次数':>14}{'保存值 MiB':>12}{'峰值 MiB':>12}"
          f"{'GPU ms':>10}{'CPU ms':>10}")
    for m in modes:
        print(f"{'重算开' if m['gradient_checkpointing'] else '重算关':<16}"
              f"{m['silu_calls']:>14}{m['saved_forward_MiB']:>12.2f}"
              f"{m['memory']['首次反向后']['max_allocated_MiB']:>12.1f}"
              f"{m['gpu_total_ms']:>10.1f}{m['cpu_total_ms']:>10.1f}")
    a, b = modes
    print(f"\n两种配置的逐 microbatch loss：{a['loss_per_microbatch']} / "
          f"{b['loss_per_microbatch']}；梯度范数 {a['grad_norm']} / {b['grad_norm']}")
    print("阶段时间（GPU / CPU，单位 ms）：")
    for m in modes:
        tag = "重算开" if m["gradient_checkpointing"] else "重算关"
        cells = "  ".join(f"{k}={m['gpu_ms'].get(k, 0):.1f}/{m['cpu_ms'].get(k, 0):.1f}"
                          for k in ("fetch", "h2d", "forward", "backward", "clip",
                                    "optimizer", "zero_grad"))
        print(f"  {tag}: {cells}")
    print("\n三种吞吐分母（同一次更新）：")
    for m in modes:
        t = m["throughput"]
        print(f"  {'重算开' if m['gradient_checkpointing'] else '重算关'}: "
              f"{t['samples']} 样本 / {t['effective_tokens']} 有效 token / "
              f"{t['optimizer_updates']} 次更新 → "
              f"{t['samples_per_s']} 样本每秒、{t['effective_tokens_per_s']} token 每秒、"
              f"{t['updates_per_s']} 次更新每秒")
    return {"experiment": "timeline", "modes": modes,
            "config": {"seq": SEQ, "micro_bs": MICRO_BS, "accum": ACCUM,
                       "workers": args.workers, "fused_optimizer": args.fused,
                       "warmup_steps": args.warmup}}


# ------------------------------------------------------------------ precision
def run_precision(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    shard = build_shard(Path(args.workdir), tokenizer, SEQ, N_SAMPLES)
    dataset = TokenShardDataset(shard, SEQ)
    gen = torch.Generator().manual_seed(0)
    inputs = {
        "分片文本": torch.stack([dataset[i] for i in range(4)]),
        "随机 token": torch.randint(0, 40000, (4, SEQ), generator=gen),
    }
    results = []
    for input_name, ids_cpu in inputs.items():
        for name, autocast_dtype, use_scaler in (
                ("FP32", None, False),
                ("FP32 + BF16 autocast", torch.bfloat16, False),
                ("FP32 + FP16 autocast + scaler", torch.float16, True)):
            torch.manual_seed(0)
            model = load_model()
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
            steps = {"n": 0}
            opt.register_step_post_hook(
                lambda *a, **k: steps.__setitem__("n", steps["n"] + 1))
            first_bad, attempts = None, []
            for attempt in range(2):
                ids = ids_cpu.to("cuda")
                labels = make_batch(ids)
                ctx = (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype
                       else torch.autocast("cuda", enabled=False))
                with ctx:
                    out = model(input_ids=ids, labels=labels)
                if use_scaler:
                    scaler.scale(out.loss).backward()
                    scaler.unscale_(opt)
                else:
                    out.loss.backward()
                bad = None
                for pname, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = {"param": pname, "shape": list(p.grad.shape),
                               "dtype": str(p.grad.dtype).replace("torch.", ""),
                               "nonfinite": int((~torch.isfinite(p.grad)).sum()),
                               "total": int(p.grad.numel())}
                        break
                first_bad = first_bad or bad
                scale_before = scaler.get_scale() if use_scaler else None
                gnorm = None
                if bad is None:
                    gnorm = round(float(torch.nn.utils.get_total_norm(
                        [p.grad for p in model.parameters() if p.grad is not None])), 6)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                attempts.append({"loss": round(float(out.loss.detach()), 7),
                                 "logits_dtype": str(out.logits.dtype).replace("torch.", ""),
                                 "grad_finite": bad is None,
                                 "grad_norm": gnorm,
                                 "first_nonfinite": bad,
                                 "scale_before": scale_before,
                                 "scale_after": scaler.get_scale() if use_scaler else None,
                                 "optimizer_calls_so_far": steps["n"]})
                opt.zero_grad(set_to_none=True)
            opt_bytes = sum(t.numel() * t.element_size()
                            for st in opt.state.values() for t in st.values()
                            if torch.is_tensor(t))
            results.append({"input": input_name, "config": name, "attempts": attempts,
                            "optimizer_calls": steps["n"],
                            "optimizer_state_MiB": round(opt_bytes / 2 ** 20, 1),
                            "first_nonfinite": first_bad})
            del model, opt
            torch.cuda.empty_cache()

    print(f"{'输入':<12}{'配置':<32}{'首次 loss':>12}{'logits':>10}"
          f"{'两次 finite':>13}{'梯度范数':>12}{'optimizer 调用':>14}{'state MiB':>11}")
    for r in results:
        fin = "/".join("是" if a["grad_finite"] else "否" for a in r["attempts"])
        gn = r["attempts"][0]["grad_norm"]
        print(f"{r['input']:<12}{r['config']:<32}{r['attempts'][0]['loss']:>12.5f}"
              f"{r['attempts'][0]['logits_dtype']:>10}{fin:>13}"
              f"{('-' if gn is None else f'{gn:.4f}'):>12}"
              f"{r['optimizer_calls']:>14}{r['optimizer_state_MiB']:>11.1f}")
    for r in results:
        if r["first_nonfinite"]:
            b = r["first_nonfinite"]
            print(f"\n{r['input']} / {r['config']}：第一个非有限梯度在 {b['param']}，"
                  f"shape={b['shape']}，dtype={b['dtype']}，"
                  f"{b['nonfinite']}/{b['total']} 个元素")
            print("  梯度以 FP32 落盘不代表它是 FP32 算出来的；已产生的 Inf 不会因转宽而恢复。")
        if r["attempts"][0]["scale_after"]:
            print(f"  {r['input']} / {r['config']} 的 scale："
                  f"{[a['scale_before'] for a in r['attempts']]} → "
                  f"{[a['scale_after'] for a in r['attempts']]}")
    # scale 扫描：溢出与否由 scale × 中间梯度量级决定，不是 dtype 本身的属性
    sweep = []
    ids_cpu = inputs["随机 token"]
    for exp in (16, 20, 24, 28):
        torch.manual_seed(0)
        model = load_model()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        scaler = torch.amp.GradScaler("cuda", init_scale=float(2 ** exp))
        ids = ids_cpu.to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, labels=make_batch(ids))
        scaler.scale(out.loss).backward()
        scaler.unscale_(opt)
        bad = None
        for pname, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad = {"param": pname, "shape": list(p.grad.shape),
                       "nonfinite": int((~torch.isfinite(p.grad)).sum()),
                       "total": int(p.grad.numel())}
                break
        before = model.lm_head.weight.detach()[0, :4].float().cpu().tolist()
        scaler.step(opt)
        scaler.update()
        after = model.lm_head.weight.detach()[0, :4].float().cpu().tolist()
        sweep.append({"init_scale": 2 ** exp, "grad_finite": bad is None,
                      "first_nonfinite": bad, "scale_after": scaler.get_scale(),
                      "params_moved": before != after})
        del model, opt
        torch.cuda.empty_cache()

    print(f"\n随机 token 输入下扫描初始 scale（FP16 autocast）：")
    print(f"  {'init_scale':>12}{'梯度 finite':>12}{'参数是否更新':>14}"
          f"{'update 后 scale':>16}  首个非有限张量")
    for r in sweep:
        b = r["first_nonfinite"]
        tag = "-" if b is None else f"{b['param']}（{b['nonfinite']}/{b['total']}）"
        print(f"  {r['init_scale']:>12}{str(r['grad_finite']):>12}"
              f"{str(r['params_moved']):>14}{r['scale_after']:>16.0f}  {tag}")
    print("  溢出与否取决于 scale × 中间梯度的量级，不是 dtype 本身的属性；")
    print("  同一份输入换一个初始 scale 就会跨过边界，scaler 的回退正是为此存在。")
    return {"experiment": "precision", "results": results, "scale_sweep": sweep}


# ------------------------------------------------------------------ compile
def run_compile(args) -> dict:
    import torch._dynamo as dynamo
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    shard = build_shard(Path(args.workdir), tokenizer, SEQ, N_SAMPLES)
    dataset = TokenShardDataset(shard, SEQ)

    def one_step(model, opt, ids):
        labels = make_batch(ids)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(out.loss.detach())

    def timed_steps(model, opt, ids, n=5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one_step(model, opt, ids)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    ids512 = torch.stack([dataset[i] for i in range(MICRO_BS)]).to("cuda")
    ids384 = ids512[:, :384].contiguous()
    ids256 = ids512[:, :256].contiguous()

    rows = []
    torch.manual_seed(0)
    model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(3):
        one_step(model, opt, ids512)
    rows.append({"case": "eager 稳态", "ms_per_step": round(timed_steps(model, opt, ids512), 2)})
    del model, opt
    torch.cuda.empty_cache()

    dynamo.reset()
    torch.manual_seed(0)
    model = load_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    compiled = torch.compile(model)
    t0 = time.perf_counter()
    one_step(compiled, opt, ids512)
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    steady = timed_steps(compiled, opt, ids512)
    counters = dict(dynamo.utils.counters["frames"])
    rows.append({"case": "compile 首次调用", "ms_per_step": round(first, 1)})
    rows.append({"case": "compile 稳态", "ms_per_step": round(steady, 2)})

    # 形状变化：观察重编译次数
    recompiles = []
    for tag, batch in (("512 再来一次", ids512), ("384", ids384), ("256", ids256),
                       ("384 第二次", ids384)):
        before = dynamo.utils.counters["frames"].get("ok", 0)
        t0 = time.perf_counter()
        one_step(compiled, opt, batch)
        torch.cuda.synchronize()
        dur = (time.perf_counter() - t0) * 1e3
        after = dynamo.utils.counters["frames"].get("ok", 0)
        recompiles.append({"shape": tag, "ms": round(dur, 1),
                           "frames_ok_delta": after - before})
    del model, opt, compiled
    torch.cuda.empty_cache()

    print(f"{'情形':<20}{'每步 ms':>12}")
    for r in rows:
        print(f"{r['case']:<20}{r['ms_per_step']:>12.2f}")
    print(f"\ndynamo frames 计数（首次编译后）：{counters}")
    print(f"{'输入形状':<16}{'该次耗时 ms':>14}{'新增编译帧':>12}")
    for r in recompiles:
        print(f"{r['shape']:<16}{r['ms']:>14.1f}{r['frames_ok_delta']:>12}")
    print("形状变化会重新进入编译；同一形状第二次出现时不再编译。")
    return {"experiment": "compile", "rows": rows, "counters": counters,
            "shape_changes": recompiles}


EXPERIMENTS = {"timeline": run_timeline, "precision": run_precision,
               "compile": run_compile}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="timeline")
    ap.add_argument("--workdir", required=True, help="学习盘上的分词分片目录")
    ap.add_argument("--outdir")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--fused", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "本 lab 需要 GPU"
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    print(f"模型 {MODEL} | 序列 {SEQ} | micro_bs {MICRO_BS} | accum {ACCUM}")
    report = EXPERIMENTS[args.experiment](args)
    report["env"] = {"torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0), "model": MODEL}
    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=False)
        (out / f"{args.experiment}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化结果写入 {out}/{args.experiment}.json")


if __name__ == "__main__":
    main()
