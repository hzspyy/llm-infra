
import json, os, subprocess, sys, time

def main():
    # 子进程自己 setsid 脱离父进程组：按进程组回收抓不到它
    # 注意：不能再调用 os.setsid()——Popen(start_new_session=True) 已经让它成为会话首进程，
    # 再次 setsid 会抛 EPERM 让子进程在写出标记之前就退出，反例本身会失效。
    code = "import os,time;open(%r,'w').write(str(os.getpid()));time.sleep(45)"
    marker = os.path.join(os.getcwd(), "escape.pid")
    subprocess.Popen([sys.executable, "-c", code % marker], start_new_session=True)
    print(json.dumps({"escaped": True}), flush=True)
    time.sleep(45)

main()
