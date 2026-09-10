import csv, sys, collections
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if l.startswith('"'))
        if r.get("Kernel Name")]
per = collections.OrderedDict()
for r in rows:
    per.setdefault(r["Kernel Name"].split("(")[0], {})[r["Metric Name"]] = r["Metric Value"]
REQ = 196608          # 6291456 次读 / 32 线程 = 每个 warp 一次请求
C = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
W = "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
T = "gpu__time_duration.sum"
print(f"  {'kernel':18s}{'耗时ns':>12s}{'bank冲突':>16s}{'smem读wavefront':>18s}{'wavefront/请求':>16s}")
for k, d in per.items():
    wf = int(d.get(W, "0").replace(",", ""))
    print(f"  {k:18s}{d.get(T,''):>12s}{d.get(C,''):>16s}{d.get(W,''):>18s}{wf/REQ:>16.2f}")
