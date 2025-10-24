
import os, time, shutil, sys
from pathlib import Path

def rm_older_than(path: str, days: int):
    now = time.time()
    cutoff = now - days*86400
    removed = 0
    for root, dirs, files in os.walk(path):
        for d in list(dirs):
            p = os.path.join(root, d)
            try:
                if os.path.getmtime(p) < cutoff:
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    return removed

if __name__ == "__main__":
    base = "artifacts"
    days = int(sys.argv[1]) if len(sys.argv)>1 else 30
    n = rm_older_than(base, days)
    print(f"Removed {n} old directories (> {days} days) under {base}.")
