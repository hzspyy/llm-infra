#!/usr/bin/env python3
"""DCP 的保存契约：同步保存、异步 staging、四种失败注入与 2→4 重新分片。

两种模式：
  single  单进程：真实 dcp.save / dcp.async_save，注入四种失败，
          再用一个只认完整 checkpoint 的 latest 选择器验证结果
  shard   torchrun 下用 DTensor 分片保存，换一个 world size 加载，
          比较分片元数据与还原后的张量

写入目录由 --workdir 指定（必须在学习盘上）。

Usage:
    python labs/L7/dcp_checkpoint_contract.py --mode single --workdir "$RUN_DIR/ckpt"
    torchrun --standalone --nproc_per_node=2 labs/L7/dcp_checkpoint_contract.py \\
        --mode shard --phase save --workdir "$RUN_DIR/ckpt"
    torchrun --standalone --nproc_per_node=4 labs/L7/dcp_checkpoint_contract.py \\
        --mode shard --phase load --workdir "$RUN_DIR/ckpt"
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ------------------------------------------------------------------ 训练状态
def make_state(step: int) -> dict:
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.randn(8, 16)
    for _ in range(max(step, 1)):          # 至少一次 step，让 m/v 存在
        opt.step()
    if step == 0:                          # step=0 表示"空状态"，用于加载目标
        with torch.no_grad():
            model.weight.zero_()
    sd = {
        "model": model.state_dict(),
        "optim": {k: v for k, v in opt.state_dict()["state"][0].items()
                  if torch.is_tensor(v)},
        "global_step": torch.tensor(step),
        "data_cursor": torch.tensor(step * 4),      # 已提交的样本数
    }
    return sd, model, opt


def checkpoint_files(path: Path) -> list[str]:
    return sorted(p.name for p in path.iterdir()) if path.exists() else []


def is_complete(path: Path) -> tuple[bool, str]:
    """latest 选择器：只有元数据齐全且引用的分片都在，才认为可用。"""
    if not (path / ".metadata").exists():
        return False, "缺少 .metadata（提交未完成）"
    try:
        md = FileSystemReader(str(path)).read_metadata()
    except Exception as exc:
        return False, f"元数据不可读：{type(exc).__name__}"
    referenced = {item.relative_path for item in md.storage_data.values()}
    missing = sorted(r for r in referenced if not (path / r).exists())
    if missing:
        return False, f"分片文件缺失：{missing}"
    return True, f"{len(md.state_dict_metadata)} 个条目，{len(referenced)} 个分片文件"


def pick_latest(root: Path) -> tuple[Path | None, list[tuple[str, bool, str]]]:
    rows = []
    best = None
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        ok, why = is_complete(d)
        rows.append((d.name, ok, why))
        if ok:
            best = d
    return best, rows


# ------------------------------------------------------------------ single
def run_single(args) -> dict:
    root = Path(args.workdir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    report: dict = {"cases": []}

    head("A 同步保存：写完就是完整的")
    sd, _, _ = make_state(step=1)
    path = root / "step-0001-sync"
    dcp.save(sd, checkpoint_id=str(path))
    print(f"  {path.name} 的文件：{checkpoint_files(path)}")
    ok, why = is_complete(path)
    print(f"  完整性：{ok}（{why}）")
    report["cases"].append({"case": "sync", "files": checkpoint_files(path), "ok": ok})

    head("B 异步保存：返回、staging 完成、文件写完是三件事")
    sd2, _, _ = make_state(step=2)
    path2 = root / "step-0002-async"
    fut = dcp.async_save(sd2, checkpoint_id=str(path2))
    print(f"  async_save 立刻返回 {type(fut).__name__}，此时目录内容："
          f"{checkpoint_files(path2)}")
    print(f"  Future 是否已完成：{fut.done()}")
    fut.result()
    print(f"  result() 之后：{checkpoint_files(path2)}，完整性={is_complete(path2)[0]}")
    print("  返回值只说明 staging 已经把张量拷出来、训练可以继续改参数；")
    print("  文件真正落盘要等 Future 完成。两者之间崩溃就会留下半个 checkpoint。")
    report["cases"].append({"case": "async", "files": checkpoint_files(path2),
                            "ok": is_complete(path2)[0]})

    head("C 四种失败注入")
    faults = []

    # 1. 保存前失败：目录根本没建立
    p = root / "step-0003-before-save"
    ok, why = is_complete(p)
    faults.append(("保存前崩溃（目录不存在）", p.name, ok, why))

    # 2. staging 之后、写文件之前失败：只有目录，没有任何文件
    p = root / "step-0004-after-staging"
    p.mkdir()
    ok, why = is_complete(p)
    faults.append(("staging 后崩溃（空目录）", p.name, ok, why))

    # 3. 文件写完前失败：删掉一个分片文件
    sd3, _, _ = make_state(step=5)
    p = root / "step-0005-partial-shard"
    dcp.save(sd3, checkpoint_id=str(p))
    victim = next(f for f in p.iterdir() if f.suffix == ".distcp")
    victim.unlink()
    ok, why = is_complete(p)
    faults.append((f"分片写入中断（删除 {victim.name}）", p.name, ok, why))

    # 4. manifest 提交前失败：分片都在，元数据没写
    sd4, _, _ = make_state(step=6)
    p = root / "step-0006-no-metadata"
    dcp.save(sd4, checkpoint_id=str(p))
    (p / ".metadata").unlink()
    ok, why = is_complete(p)
    faults.append(("提交前崩溃（删除 .metadata）", p.name, ok, why))

    print(f"  {'注入的失败':<32}{'目录':<24}{'可用':<6}原因")
    for name, dirname, ok, why in faults:
        print(f"  {name:<32}{dirname:<24}{str(ok):<6}{why}")
    report["faults"] = [{"fault": n, "dir": d, "usable": o, "reason": w}
                        for n, d, o, w in faults]

    head("D 真正加载残缺 checkpoint 会发生什么")
    from torch.distributed.checkpoint.api import CheckpointException
    print(f"  注意 CheckpointException 的基类是 "
          f"{CheckpointException.__mro__[1].__name__}，不是 Exception——"
          f"只写 except Exception 抓不到它。")
    for name in ("step-0005-partial-shard", "step-0006-no-metadata"):
        target, _, _ = make_state(step=0)
        try:
            dcp.load(target, checkpoint_id=str(root / name))
            msg = "加载成功（说明这份残缺没有被发现）"
        except BaseException as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
        print(f"  加载 {name}：{msg}")
        report.setdefault("load_errors", []).append({"dir": name, "error": msg})

    head("E latest 选择器只认完整的 checkpoint")
    best, rows = pick_latest(root)
    for name, ok, why in rows:
        print(f"  {name:<26}{'可用' if ok else '拒绝':<6}{why}")
    print(f"\n  选中的 latest：{best.name if best else '无'}")
    print("  按 mtime 或按目录名取最大都会选中残缺目录；判据必须是"
          "元数据存在且它引用的分片文件都在。")
    report["latest"] = best.name if best else None

    head("F 恢复之后状态对不对")
    target, model, opt = make_state(step=0)
    dcp.load(target, checkpoint_id=str(root / "step-0002-async"))
    ref, _, _ = make_state(step=2)
    same = all(torch.equal(target["model"][k], ref["model"][k]) for k in ref["model"])
    print(f"  model 张量一致：{same}")
    print(f"  optimizer 状态一致："
          f"{all(torch.equal(target['optim'][k], ref['optim'][k]) for k in ref['optim'])}")
    print(f"  global_step={int(target['global_step'])}  "
          f"data_cursor={int(target['data_cursor'])}")
    print("  参数、optimizer 状态、step 与数据游标必须来自同一次保存；"
          "任何一项单独恢复都会让训练轨迹与中断前不同。")
    return report


# ------------------------------------------------------------------ shard
def run_shard(args) -> dict:
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor, init_device_mesh
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (world,))
    root = Path(args.workdir) / "sharded"

    full = torch.arange(64.0).reshape(8, 8)
    dt = distribute_tensor(full, mesh, [Shard(0)])
    local = dt.to_local()

    if args.phase == "save":
        if rank == 0 and root.exists():
            shutil.rmtree(root)
        dist.barrier()
        dcp.save({"w": dt, "step": torch.tensor(7)}, checkpoint_id=str(root))
        dist.barrier()
        if rank == 0:
            md = FileSystemReader(str(root)).read_metadata()
            chunks = md.state_dict_metadata["w"].chunks
            print(f"[save] world={world}：本 rank 的 local shape={tuple(local.shape)}")
            print(f"  文件：{checkpoint_files(root)}")
            print(f"  元数据里的 {len(chunks)} 个 chunk：")
            for c in chunks:
                print(f"    offsets={tuple(c.offsets)} sizes={tuple(c.sizes)}")
        result = {"phase": "save", "world": world,
                  "local_shape": list(local.shape)}
    else:
        empty = distribute_tensor(torch.zeros(8, 8), mesh, [Shard(0)])
        state = {"w": empty, "step": torch.tensor(0)}
        dcp.load(state, checkpoint_id=str(root))
        got = state["w"].full_tensor()
        ok = torch.equal(got, full)
        if rank == 0:
            md = FileSystemReader(str(root)).read_metadata()
            print(f"[load] world={world}：每 rank 现在持有 "
                  f"{tuple(state['w'].to_local().shape)}，"
                  f"保存时的 chunk 数={len(md.state_dict_metadata['w'].chunks)}")
            print(f"  还原出的完整张量与原始张量相等：{ok}；step={int(state['step'])}")
            print("  DCP 保存的是逻辑张量加上每片的 offsets/sizes，不是各 rank 的文件快照；")
            print("  因此换一个 world size 只是换一组读取区间，不需要先合并再切分。")
        result = {"phase": "load", "world": world,
                  "local_shape": list(state["w"].to_local().shape), "equal": bool(ok)}
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "shard"], default="single")
    ap.add_argument("--phase", choices=["save", "load"], default="save")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--outdir")
    args = ap.parse_args()
    print(f"torch {torch.__version__}")
    report = run_single(args) if args.mode == "single" else run_shard(args)
    if args.outdir and int(os.environ.get("RANK", 0)) == 0:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        name = args.mode if args.mode == "single" else f"shard_{args.phase}_{os.environ.get('WORLD_SIZE')}"
        (out / f"{name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
