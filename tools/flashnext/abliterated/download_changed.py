"""Downloads the changed tensors listed by find_changed.py (raw BF16) with HTTP range requests: resumable, 32 parallel ranges.
Usage: python download_changed.py <delta.json> <delta dir>

Files on Hugging Face's Xet storage that share most bytes with other uploads download slowly as whole files (the
client fetches far more than it keeps); fetching only the changed byte ranges avoids that."""
import json, os, sys, time, urllib.request, concurrent.futures as cf, threading
spec = json.load(open(sys.argv[1])); dest = sys.argv[2]
os.makedirs(dest, exist_ok=True)
CH = 32 << 20
done_path = os.path.join(dest, "done.txt")
done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
lock = threading.Lock()
jobs = []
for t in spec:
    p = os.path.join(dest, t["name"] + ".bf16")
    if not os.path.exists(p) or os.path.getsize(p) != t["size"]:
        with open(p, "ab") as f: f.truncate(t["size"])  # on Windows this zero-fills the file once (a few minutes for 82 GB)
    for s in range(0, t["size"], CH):
        key = f"{t['name']}@{s}"
        if key not in done: jobs.append((t, p, s, min(CH, t["size"] - s), key))
total = sum(j[3] for j in jobs); got = 0; t0 = time.time()
print(f"{len(jobs)} ranges, {total/1e9:.2f} GB to fetch", flush=True)
def work(j):
    t, p, s, n, key = j
    for attempt in range(8):
        try:
            data = urllib.request.urlopen(urllib.request.Request(t["url"], headers={"Range": f"bytes={t['off']+s}-{t['off']+s+n-1}"}), timeout=600).read()
            if len(data) != n: raise IOError(f"short read {len(data)} of {n}")
            with open(p, "r+b") as f: f.seek(s); f.write(data)
            return key, n
        except Exception as e:
            time.sleep(5 * (attempt + 1)); err = e
    raise RuntimeError(f"{key}: {err}")
with cf.ThreadPoolExecutor(32) as ex, open(done_path, "a") as log:
    for key, n in ex.map(work, jobs):
        with lock:
            log.write(key + "\n"); log.flush(); got += n
            if int(got / total * 100) != int((got - n) / total * 100):
                el = time.time() - t0
                print(f"{got/1e9:7.2f} / {total/1e9:.2f} GB, {got/el/1e6:.1f} MB/s, ETA {(total-got)/(got/el)/60:.0f} min", flush=True)
print("DONE", flush=True)
