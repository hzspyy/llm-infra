#!/usr/bin/env python3
"""从已保存的原始字节流重算 5.11 的派生统计，不重跑实验。

    python summarize_api_layer.py results/crater/api/20260912-0325/data
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mini_sse import SSEDecoder  # noqa: E402


def dechunk(raw: bytes) -> tuple[bytes, list[int]]:
    out, sizes, pos = bytearray(), [], 0
    while True:
        eol = raw.find(b"\r\n", pos)
        if eol < 0:
            break
        field = raw[pos:eol].split(b";")[0].strip()
        if not field:
            break
        try:
            size = int(field, 16)
        except ValueError:
            break
        pos = eol + 2
        if size == 0:
            break
        out += raw[pos:pos + size]
        sizes.append(size)
        pos += size + 2
    return bytes(out), sizes


def analyse(path: Path) -> dict:
    raw = path.read_bytes()
    sep = raw.find(b"\r\n\r\n")
    header = raw[:sep].decode("latin1")
    body, sizes = dechunk(raw[sep + 4:])
    parser = SSEDecoder()
    events = parser.feed(body, final=True)
    data_lines = [l for l in body.split(b"\n") if l.startswith(b"data:")]
    empty_data = [l for l in data_lines if l.strip() == b"data:"]
    comments = [l for l in body.split(b"\n") if l.startswith(b":")]
    done = [e for e in events if e["data"] == "[DONE]"]
    deltas = [e for e in events if e["data"] != "[DONE]"]
    lengths = [len(e["data"]) for e in deltas]
    return {
        "file": path.name,
        "header": header,
        "transfer_encoding_chunked": "transfer-encoding: chunked" in header.lower(),
        "chunked_frames": len(sizes),
        "chunk_size_min": min(sizes) if sizes else None,
        "chunk_size_max": max(sizes) if sizes else None,
        "body_bytes": len(body),
        "sse_events": len(events),
        "sse_delta_events": len(deltas),
        "sse_done_events": len(done),
        "keepalive_comments": len(comments),
        "empty_data_lines": len(empty_data),
        "delta_chars_min": min(lengths) if lengths else None,
        "delta_chars_max": max(lengths) if lengths else None,
        "delta_chars_median": statistics.median(lengths) if lengths else None,
        "bytes_per_delta_event": round(len(body) / len(deltas), 2) if deltas else None,
        "first_delta": deltas[0]["data"][:120] if deltas else None,
        "last_delta": deltas[-1]["data"][:120] if deltas else None,
    }


def main():
    root = Path(sys.argv[1])
    out = {"dir": str(root), "streams": []}
    for name in sorted(p.name for p in root.glob("*.bin")):
        out["streams"].append(analyse(root / name))
    audit = root / "audit.json"
    if audit.exists():
        a = json.loads(audit.read_text())
        out["layers"] = a.get("layers", {}).get("derived")
        out["tokenize_points"] = a.get("tokenize", {}).get("points")
        out["concurrency"] = a.get("concurrency")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
