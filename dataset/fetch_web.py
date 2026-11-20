"""抓维基百科中文正文，当「网上的文章」用。

本机直连维基不可达（curl 返回 000），走本机 Clash 的混合端口 127.0.0.1:7897。
Python 直连 HTTPS 不稳，统一 subprocess 调 curl.exe（Schannel + --ssl-no-revoke）。
"""

import json
import os
import subprocess
import time
import urllib.parse
import paths

OUT = rpaths.sample("web_wiki.json")
PROXY = os.environ.get("CLASH_PROXY", "http://127.0.0.1:7897")

# (维基条目, domains.py 里对应的类名)
TARGETS = [
    ("中国人民银行", "金融_经济"),
    ("中华人民共和国民法典", "法律_司法"),
    ("糖尿病", "医学_健康_心理_中医"),
    ("公路自行车", "体育"),
    ("太阳能光伏", "电力能源"),
    ("高速铁路", "交通运输"),
    ("酒店", "住宿_餐饮_酒店"),
    ("中国航天", "航空航天"),
    ("水稻", "农林牧渔"),
    ("深度学习", "人工智能_机器学习"),
    ("电子游戏", "游戏"),
    ("原油", "石油化工"),
    ("心理学", "医学_健康_心理_中医"),
    ("建筑", "房地产_建筑"),
    ("足球", "体育"),
    ("汽车", "汽车"),
    ("微处理器", "计算机_通信"),
    ("中华人民共和国刑法", "法律_司法"),
]

API = ("https://zh.wikipedia.org/w/api.php?action=query&prop=extracts"
       "&explaintext=1&exchars=1200&format=json&redirects=1&titles=")


def curl(url, timeout=40):
    r = subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-s", "-L", "-x", PROXY,
         "--max-time", str(timeout), url],
        capture_output=True, timeout=timeout + 20,
    )
    return r.stdout or b""


def fetch(title, attempts=3):
    last = "unknown"
    for i in range(attempts):
        text, err = _fetch_once(title)
        if text:
            return text, None
        last = err
        time.sleep(1.5 + i * 1.5)
    return None, last


def _fetch_once(title):
    url = API + urllib.parse.quote(title)
    raw = curl(url)
    if not raw:
        return None, "empty response (proxy?)"
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except Exception as e:
        return None, "bad json (%d bytes)" % len(raw)
    pages = d.get("query", {}).get("pages", {})
    if not pages:
        return None, "no pages"
    pg = list(pages.values())[0]
    if "missing" in pg:
        return None, "missing"
    text = " ".join((pg.get("extract") or "").split())
    text = " ".join(seg for seg in text.split(" ") if not seg.startswith("=="))
    if len(text) < 400:
        return None, "too short (%d)" % len(text)
    return text, None


def main():
    rows = []
    if os.path.isfile(OUT):
        try:
            rows = json.load(open(OUT, encoding="utf-8")).get("items", [])
        except Exception:
            rows = []
    have = {r["title"] for r in rows}
    for title, dom in TARGETS:
        if title in have:
            print("SKIP %-20s 已有" % title)
            continue
        text, err = fetch(title)
        if text:
            rows.append({"title": title, "domain": dom, "text": text})
            print("OK   %-20s %-20s %5d 字" % (title, dom, len(text)))
        else:
            print("FAIL %-20s %s" % (title, err))
    doc = {"source": "zh.wikipedia.org 正文摘录（经本机 Clash 代理）", "items": rows}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("\n成功 %d / %d，写入 %s" % (len(rows), len(TARGETS), OUT))


if __name__ == "__main__":
    main()
