
import json, os, sys, time

def main():
    spec = json.loads(sys.argv[1])
    if spec.get("sleep_ms"):
        time.sleep(spec["sleep_ms"] / 1000.0)
    if spec.get("alloc_mb"):
        blob = bytearray(spec["alloc_mb"] * 1024 * 1024)
        blob[0] = 1
    if spec.get("output_bytes"):
        sys.stdout.write("x" * spec["output_bytes"])
        sys.stdout.flush()
    if spec.get("spawn_children"):
        import subprocess
        kids = [subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
                for _ in range(spec["spawn_children"])]
        print(json.dumps({"children": [k.pid for k in kids]}), flush=True)
        time.sleep(spec.get("hold_s", 30))
    if spec.get("write_mb"):
        with open(os.path.join(os.getcwd(), "blob.bin"), "wb") as fh:
            fh.write(b"y" * (spec["write_mb"] * 1024 * 1024))
    print(json.dumps({"ok": True, "pid": os.getpid(), "cwd": os.getcwd(),
                      "uid": os.getuid(), "argv": spec}), flush=True)

main()
