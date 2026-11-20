"""快速筛出 IndustryCorpus2 里「能读」的分片。

发现：这批 parquet 有一部分是写入时没写 footer 的残片——文件以 PAR1 开头、
大小和官方 API 完全一致，但结尾没有 PAR1，pyarrow 打不开。
ModelScope 官方源和 hf-mirror 上的字节完全一样，不是镜像的锅。

标准 parquet 末尾 4 字节一定是 PAR1。于是用 HTTP Range 只取文件最后 4 字节，
几 KB 流量就能判断一个分片能不能读，不用把几百 MB 下下来再发现是坏的。

输入：$IC2_TREE_DIR（默认 $PRISM_HOME/ic2_trees）下的 ic2min_<key>.json 文件树
输出：$PRISM_HOME/shards_ok.json
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

TREE_DIR = os.environ.get("IC2_TREE_DIR") or paths.path("ic2_trees")
OUT = paths.path("shards_ok.json")
MIRROR = "https://hf-mirror.com/datasets/BAAI/IndustryCorpus2_{repo}/resolve/main/{path}"
MAX_MB = 500
PER_CLASS_CANDIDATES = 12

KEYS = {
    "transportation": "transportation",
    "medicine": "medicine_health_psychology_traditional_chinese_medicine",
    "math": "mathematics_statistics",
    "government": "current_affairs_government_administration",
    "safety": "fire_safety_food_safety",
    "petrochemical": "petrochemical",
    "computer": "computer_communication",
    "ai": "artificial_intelligence_machine_learning",
    "infosec": "other_information_services_information_security",
    "education": "subject_education_education",
    "literature": "literature_emotion",
    "water": "water_resources_ocean",
    "game": "game",
    "research": "technology_scientific_research",
    "mining": "mining",
    "hotel": "accommodation_catering_hotel",
    "manufacturing": "other_manufacturing",
    "film": "film_entertainment",
    "news": "news_media",
    "automobile": "automobile",
    "biomedicine": "biomedicine",
    "aerospace": "aerospace",
    "finance": "finance_economics",
    "sports": "sports",
    "agriculture": "agriculture_forestry_animal_husbandry_fishery",
    "realestate": "real_estate_construction",
    "tourism": "tourism_geography",
    "law": "law_judiciary",
    "energy": "electric_power_energy",
    "code": "computer_programming_code",
}

# 中文领域名（和 domains.py 一致）
ZH = {
    "transportation": "交通运输", "medicine": "医学_健康_心理_中医",
    "math": "数学_统计学", "government": "时政_政务_行政",
    "safety": "消防安全_食品安全", "petrochemical": "石油化工",
    "computer": "计算机_通信", "ai": "人工智能_机器学习",
    "infosec": "其他信息服务_信息安全", "education": "学科教育_教育",
    "literature": "文学_情感", "water": "水利_海洋", "game": "游戏",
    "research": "科技_科学研究", "mining": "采矿",
    "hotel": "住宿_餐饮_酒店", "manufacturing": "其他制造",
    "film": "影视_娱乐", "news": "新闻传媒", "automobile": "汽车",
    "biomedicine": "生物医药", "aerospace": "航空航天",
    "finance": "金融_经济", "sports": "体育",
    "agriculture": "农林牧渔", "realestate": "房地产_建筑",
    "tourism": "旅游_地理", "law": "法律_司法",
    "energy": "电力能源", "code": "计算机编程_代码",
}


def probe(key, repo, path, size):
    tmp = os.path.join(tempfile.gettempdir(), "ic2probe_%d.bin" % (abs(hash((key, path))) % 10**9))
    url = MIRROR.format(repo=repo, path=path)
    try:
        subprocess.run(
            ["curl.exe", "--ssl-no-revoke", "-s", "-L", "--max-time", "20",
             "-r", "-4", "-o", tmp, url],
            capture_output=True, timeout=25,
        )
        data = open(tmp, "rb").read()
        ok = data == b"PAR1"
    except Exception:
        ok = False
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)
    return key, repo, path, size, ok


def main():
    jobs = []
    for key, repo in KEYS.items():
        p = os.path.join(TREE_DIR, "ic2min_%s.json" % key)
        if not os.path.isfile(p):
            print("缺文件树: %s" % key)
            continue
        files = [x for x in json.load(open(p, encoding="utf-8"))
                 if x["type"] == "file" and x["size"] <= MAX_MB * 1024**2]
        files.sort(key=lambda x: x["size"])
        for f in files[:PER_CLASS_CANDIDATES]:
            jobs.append((key, repo, f["path"], f["size"]))
    print("探测 %d 个候选分片的尾部 4 字节 ..." % len(jobs), flush=True)

    t0 = time.time()
    good = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(probe, *j) for j in jobs]
        for fut in as_completed(futs):
            key, repo, path, size, ok = fut.result()
            if ok and (key not in good or size < good[key]["size"]):
                good[key] = {"zh": ZH[key], "repo": repo, "path": path,
                             "size": size, "size_mb": round(size / 1024**2, 1)}
    print("耗时 %.1f 秒，%d/%d 个行业找到可读分片" %
          (time.time() - t0, len(good), len(KEYS)), flush=True)
    for key in sorted(good, key=lambda k: good[k]["size"]):
        g = good[key]
        print("  %-22s %7.1f MB  %s" % (g["zh"], g["size_mb"], g["path"]), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False, indent=2)
    print("已写入 %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
