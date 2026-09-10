#!/usr/bin/env python3
"""L2.2 lab · 把 `-Xptxas -v` 的输出解析成一张表。

ptxas 的四个数字决定了一个 kernel 的命运：
  registers    → 每 SM 能驻留多少线程（65536 / registers）
  stack frame  → 有没有用到 local memory（在显存里！）
  spill stores/loads → 寄存器溢出量。**不为 0 就要警惕**

用法：nvcc ... -Xptxas -v 2> log.txt && python parse_ptxas.py log.txt
"""
import re
import sys

REG_FILE = 65536          # 每 SM 的 32 位寄存器数（Blackwell 消费级）


def main() -> None:
    txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    blocks = re.split(r"ptxas info\s+: Compiling entry function ", txt)
    hdr = ("kernel", "寄存器", "栈帧B", "spill存B", "spill取B", "每SM线程上限")
    print("  %-30s %8s %8s %10s %10s %14s" % hdr)
    for b in blocks[1:]:
        name = re.search(r"^'([^']+)'", b)
        reg = re.search(r"Used (\d+) registers", b)
        if not (name and reg):
            continue
        st = re.search(r"(\d+) bytes stack frame", b)
        ss = re.search(r"(\d+) bytes spill stores", b)
        sl = re.search(r"(\d+) bytes spill loads", b)
        r = int(reg.group(1))
        warn = ""
        if ss and int(ss.group(1)) > 0:
            warn = "   ← 溢出到 local memory（显存！）"
        print("  %-30s %8d %8s %10s %10s %14d%s" % (
            name.group(1)[:30], r,
            st.group(1) if st else "?", ss.group(1) if ss else "?",
            sl.group(1) if sl else "?", REG_FILE // r, warn))


if __name__ == "__main__":
    main()
