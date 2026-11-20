"""从 hf-mirror 拉 IndustryCorpus2 的小分片，转成 prepare.py 能读的目录结构。

为什么是「小分片」：IndustryCorpus2 中文全量约 1TB，单个 parquet 常有 800MB。
这里只取每个行业子仓库里最小的那一个分片，并按字符数截断，
凑一份几百 MB 的真语料，够本机跑一整晚，也不会把盘撑爆。

输出：
  raw/<英文仓库名>.parquet          原分片
  corpus/<中文领域名>/part.jsonl    每行 {"text": ...}

下载走 curl.exe（本机 Python 的 HTTPS 直连不稳），8 并发。

注意：这是初版，只按「最小分片」下载，**没有校验 parquet 是否完整**。
实测这批分片里有 11/16 是缺 footer 的残片。
请改用 scan_ic2.py 先筛可读分片，再用 fetch_ok.py 下载；
本文件只保留作 30 个行业最小分片的路径清单。
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import paths

MIRROR = "https://hf-mirror.com/datasets/BAAI/IndustryCorpus2_{repo}/resolve/main/{path}"

# (中文领域名, 子仓库名, 最小分片路径, 该分片 MB 数)
# 中文名必须和 domains.py 完全一致，prepare.py 靠目录名认领域。
SHARDS = [
    ("消防安全_食品安全", "fire_safety_food_safety", "chinese/middle/rank_00473.parquet", 13.7),
    ("学科教育_教育", "subject_education_education", "chinese/middle/rank_00656.parquet", 16.2),
    ("数学_统计学", "mathematics_statistics", "chinese/low/rank_00237.parquet", 25.2),
    ("生物医药", "biomedicine", "chinese/high/rank_00015.parquet", 25.5),
    ("计算机_通信", "computer_communication", "chinese/high/rank_00020.parquet", 25.7),
    ("其他信息服务_信息安全", "other_information_services_information_security", "chinese/low/rank_00249.parquet", 27.0),
    ("汽车", "automobile", "chinese/high/rank_00014.parquet", 27.3),
    ("人工智能_机器学习", "artificial_intelligence_machine_learning", "chinese/high/rank_00011.parquet", 28.2),
    ("农林牧渔", "agriculture_forestry_animal_husbandry_fishery", "chinese/middle/rank_00324.parquet", 29.8),
    ("房地产_建筑", "real_estate_construction", "chinese/low/rank_00267.parquet", 30.6),
    ("体育", "sports", "chinese/middle/rank_00627.parquet", 31.7),
    ("住宿_餐饮_酒店", "accommodation_catering_hotel", "chinese/low/rank_00137.parquet", 32.2),
    ("影视_娱乐", "film_entertainment", "chinese/middle/rank_00424.parquet", 32.6),
    ("文学_情感", "literature_emotion", "chinese/middle/rank_00526.parquet", 41.8),
    ("法律_司法", "law_judiciary", "chinese/middle/rank_00495.parquet", 51.3),
    ("旅游_地理", "tourism_geography", "chinese/low/rank_00297.parquet", 61.0),
    ("金融_经济", "finance_economics", "chinese/high/rank_00055.parquet", 76.8),
    ("计算机编程_代码", "computer_programming_code", "chinese/low/rank_00163.parquet", 83.0),
    ("科技_科学研究", "technology_scientific_research", "chinese/low/rank_00292.parquet", 95.9),
    ("医学_健康_心理_中医", "medicine_health_psychology_traditional_chinese_medicine", "chinese/high/rank_00080.parquet", 126.7),
    ("其他制造", "other_manufacturing", "chinese/middle/rank_00584.parquet", 151.8),
    ("交通运输", "transportation", "chinese/middle/rank_00721.parquet", 168.4),
    ("游戏", "game", "chinese/high/rank_00057.parquet", 177.7),
    ("水利_海洋", "water_resources_ocean", "chinese/middle/rank_00725.parquet", 198.4),
    ("石油化工", "petrochemical", "chinese/high/rank_00091.parquet", 304.9),
    ("新闻传媒", "news_media", "chinese/high/rank_00082.parquet", 318.3),
    ("采矿", "mining", "chinese/low/rank_00247.parquet", 334.0),
    ("时政_政务_行政", "current_affairs_government_administration", "chinese/high/rank_00036.parquet", 404.4),
    ("电力能源", "electric_power_energy", "chinese/high/rank_00041.parquet", 410.9),
    ("航空航天", "aerospace", "chinese/low/rank_00140.parquet", 413.7),
]


def download(item, raw_dir, timeout):
    zh, repo, path, mb = item
    out = os.path.join(raw_dir, repo + ".parquet")
    if os.path.isfile(out) and os.path.getsize(out) > 100000:
        return zh, "cached", os.path.getsize(out)
    url = MIRROR.format(repo=repo, path=path)
    cmd = ["curl.exe", "--ssl-no-revoke", "-s", "-L", "--fail",
           "--max-time", str(timeout), "-o", out, "-w", "%{http_code}", url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    code = (p.stdout or "").strip()
    size = os.path.getsize(out) if os.path.isfile(out) else 0
    if code != "200" or size < 100000:
        if os.path.isfile(out):
            os.remove(out)
        return zh, "FAIL http=%s (%s)" % (code, (p.stderr or "").strip()[:80]), size
    return zh, "ok", size


def to_corpus(item, raw_dir, corpus_dir, max_chars):
    import pandas as pd

    zh, repo, path, mb = item
    src = os.path.join(raw_dir, repo + ".parquet")
    if not os.path.isfile(src):
        return None
    df = pd.read_parquet(src, columns=["text"])
    d = os.path.join(corpus_dir, zh)
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
            if total >= max_chars:
                break
    return zh, n, total


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=16, help="取最小的 N 个行业分片；0 表示全部 30 个")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-domain-chars", type=int, default=200000, help="每个领域截多少字")
    ap.add_argument("--raw-dir", default=paths.path("raw"))
    ap.add_argument("--corpus-dir", default=paths.path("corpus"))
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    items = sorted(SHARDS, key=lambda x: x[3])
    if args.top > 0:
        items = items[:args.top]
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.corpus_dir, exist_ok=True)

    print("准备下载 %d 个行业分片，合计约 %.0f MB" %
          (len(items), sum(x[3] for x in items)), flush=True)
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, it, args.raw_dir, args.timeout): it for it in items}
        for fut in as_completed(futs):
            zh, status, size = fut.result()
            print("  %-24s %-10s %.1f MB" % (zh, status, size / 1024**2), flush=True)
            if status in ("ok", "cached"):
                ok += 1
    print("下载完成 %d/%d，耗时 %.1f 秒" % (ok, len(items), time.time() - t0), flush=True)

    print("转成按领域分目录的 jsonl ...", flush=True)
    rows, chars = 0, 0
    for it in items:
        r = to_corpus(it, args.raw_dir, args.corpus_dir, args.per_domain_chars)
        if r:
            zh, n, c = r
            print("  %-24s %6d 条 / %d 字" % (zh, n, c), flush=True)
            rows += n
            chars += c
    print("合计 %d 条 / %d 字 -> %s" % (rows, chars, args.corpus_dir), flush=True)
    print("下一步：prepare.py --input-dir \"%s\" --vocab-size 16000 "
          "--out \"$PRISM_HOME/blocks.pt\"" % args.corpus_dir, flush=True)


if __name__ == "__main__":
    main()
