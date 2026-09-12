#!/usr/bin/env python3
"""Read-only analysis of immutable L5.10 raw files; JSON to stdout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
a = p.parse_args()
root = a.root
rows = []
for n in [1, 8, 9, 16, 64]:
    events = json.loads((root / f"routing-{n}.json").read_text())
    mappings = [e for e in events if e["kind"] == "mapping"]
    changed = [e for e in events if e["kind"] == "activate" and e["before"] != e["after"]]
    unchanged = [e for e in events if e["kind"] == "activate" and e["before"] == e["after"]]
    assert max(len(e["counts"]) for e in mappings) <= 8
    transitions = []
    for e in mappings:
        ids = sorted(map(int, e["counts"]))
        if not transitions or ids != transitions[-1]["ids"]:
            transitions.append({"ids": ids, "tokens": e["tokens"], "counts": e["counts"]})
    rows.append({"adapters": n, "mapping_calls": len(mappings),
        "shape_histogram": dict(Counter(e["tokens"] for e in mappings)),
        "unique_adapter_histogram": dict(Counter(len(e["counts"]) for e in mappings)),
        "slot_changes": len(changed), "slot_hits": len(unchanged),
        "changed_median_ms_sync": statistics.median(e["wall_ms_sync"] for e in changed) if changed else None,
        "hit_median_ms_sync": statistics.median(e["wall_ms_sync"] for e in unchanged) if unchanged else None,
        "transitions": transitions})
kernels = json.loads((root / "kernel_summary.json").read_text())
trace = json.loads((root / "mixed8.trace.json").read_text())
trace_kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
trace_counts = Counter(e["name"] for e in trace_kernels)
for k in kernels:
    if "lora" in k["name"].lower():
        assert trace_counts[k["name"]] == k["calls"], k
summary = {"routing": rows, "kernel_cuda_us_sum": sum(k["cuda_us"] for k in kernels),
    "lora_cuda_us_sum": sum(k["cuda_us"] for k in kernels if "lora" in k["name"].lower()),
    "trace_kernel_count": len(trace_kernels), "lora_trace_counts": {k: v for k, v in trace_counts.items() if "lora" in k.lower()},
    "trace_sha256": hashlib.sha256((root / "mixed8.trace.json").read_bytes()).hexdigest()}
print(json.dumps(summary, indent=2))
