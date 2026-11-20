"""按 scan_ic2.py 筛出的「可读分片清单」下载并转成训练语料。

只下载 <= 100MB 的分片（其余留着以后补），下载后逐个校验 parquet 魔数，
残片直接丢弃，不会污染 corpus。
"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import paths

ROOT = "$PRISM_HOME"
SHARDS_JSON = os.path.join(ROOT, "shards_ok.json")
RAW = os.path.join(ROOT, "raw")
CORPUS = os.path.join(ROOT, "corpus")
MIRROR = "https://hf-mirror.com/datasets/BAAI/IndustryCorpus2_{repo}/resolve/main/{path}"
MAX_MB = 500
PER_DOMAIN_CHARS = 12000000
WORKERS = 6


def valid(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 100000:
        return False
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, 2)
        tail = f.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def download(item):
    key, g = item
    out = os.path.join(RAW, g["repo"] + ".parquet")
    if valid(out):
        return g["zh"], "cached", os.path.getsize(out)
    url = MIRROR.format(repo=g["repo"], path=g["path"])
    subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-s", "-L", "--max-time", "900",
         "-o", out, url],
        capture_output=True, text=True,
    )
    if not valid(out):
        if os.path.isfile(out):
            os.remove(out)
        return g["zh"], "BAD", 0
    return g["zh"], "ok", os.path.getsize(out)


def to_corpus(g):
    import pandas as pd

    src = os.path.join(RAW, g["repo"] + ".parquet")
    if not valid(src):
        return None
    df = pd.read_parquet(src, columns=["text"])
    d = os.path.join(CORPUS, g["zh"])
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "part.jsonl")
    total, n = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for t in df["text"]:
            t = str(t).strip()
            if not t:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total += len(t)
            n += 1
            if total >= PER_DOMAIN_CHARS:
                break
    return g["zh"], n, total


def main():
    shards = json.load(open(SHARDS_JSON, encoding="utf-8"))
    picked = {k: g for k, g in shards.items() if g["size_mb"] <= MAX_MB}
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(CORPUS, exist_ok=True)

    print("挑选 %d 个行业（分片 <= %dMB），合计约 %.0f MB" %
          (len(picked), MAX_MB, sum(g["size_mb"] for g in picked.values())), flush=True)
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download, it) for it in picked.items()]
        for fut in as_completed(futs):
            zh, status, size = fut.result()
            print("  %-22s %-8s %.1f MB" % (zh, status, size / 1024**2), flush=True)
            if status in ("ok", "cached"):
                ok += 1
    print("下载完成 %d/%d，耗时 %.1f 秒" % (ok, len(picked), time.time() - t0), flush=True)

    print("转成按领域分目录的 jsonl ...", flush=True)
    usable = [g for g in shards.values()
              if valid(os.path.join(RAW, g["repo"] + ".parquet"))]
    rows, chars = 0, 0
    for g in usable:
        r = to_corpus(g)
        if r:
            print("  %-22s %6d 条 / %d 字" % r, flush=True)
            rows += r[1]
            chars += r[2]
    print("合计 %d 个领域 / %d 条 / %d 字 -> %s" % (len(usable), rows, chars, CORPUS), flush=True)


if __name__ == "__main__":
    main()
