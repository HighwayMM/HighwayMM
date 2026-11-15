"""Minimal torchrun replacement for native Windows.

torch 2.5's torchrun static rendezvous builds a libuv TCPStore regardless of USE_LIBUV=0 and crashes on
Windows ("PyTorch was built without libuv support"). Workers' env:// init does honour USE_LIBUV, so we
just spawn one process per GPU with the torchrun env contract (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_*).
If any rank fails, the others are terminated and the exit code is propagated.

  python code\\train\\spawn.py --nproc 3 [--port 29501] script.py [script args...]
"""
import os
import random
import subprocess
import sys
import time


def main():
    argv = sys.argv[1:]
    nproc, port = 1, random.randint(29500, 29999)
    while argv and argv[0].startswith("--"):
        if argv[0] == "--nproc":
            nproc = int(argv[1]); argv = argv[2:]
        elif argv[0] == "--port":
            port = int(argv[1]); argv = argv[2:]
        else:
            break
    if not argv:
        sys.exit("usage: spawn.py --nproc N script.py [args]")
    procs = []
    for r in range(nproc):
        env = dict(os.environ, RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(nproc), LOCAL_WORLD_SIZE=str(nproc),
                   GROUP_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
        procs.append(subprocess.Popen([sys.executable, "-u"] + argv, env=env))
    code = 0
    try:
        while True:
            states = [p.poll() for p in procs]
            bad = [c for c in states if c not in (None, 0)]
            if bad:
                code = bad[0]
                print(f"[spawn] a rank exited with {code}; terminating the others", file=sys.stderr)
                break
            if all(c == 0 for c in states):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        code = 130
    finally:
        for p in procs:
            if p.poll() is None:
                if code == 130:
                    time.sleep(3)  # let ranks record status=aborted
                p.terminate()
    sys.exit(code)


if __name__ == "__main__":
    main()
