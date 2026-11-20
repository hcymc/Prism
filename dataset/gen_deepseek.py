"""让 DeepSeek 按领域写文章，用来测模型在「非语料文风」上的表现。

每个类目给一个具体题目，要求不许出现类目名、不许分点、写成一整段。
API key 从环境变量 DEEPSEEK_API_KEY 读，不落盘、不打印。
本机 Python 直连 HTTPS 不稳，统一走 subprocess 调 curl.exe（Schannel + --ssl-no-revoke）。

  python dataset/gen_deepseek.py --limit 2      # 先试两条
  python dataset/gen_deepseek.py                # 跑全部
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import paths

OUT = rpaths.sample("llm_articles.jsonl")
URL = "https://api.deepseek.com/chat/completions"

# (类目, 题目, 附加要求)
TOPICS = [
    ("交通运输", "城市地铁早高峰的调度安排", ""),
    ("医学_健康_心理_中医", "术后切口感染的观察与处理", ""),
    ("数学_统计学", "用中心极限定理解释抽样误差为什么随样本量下降", ""),
    ("时政_政务_行政", "某市推行政务服务一网通办的举措", ""),
    ("消防安全_食品安全", "大型商超的消防通道检查与冷链食品抽检", ""),
    ("石油化工", "催化裂化装置的开工步骤与日常巡检", ""),
    ("计算机_通信", "5G 基站的光模块选型与链路预算", ""),
    ("人工智能_机器学习", "用注意力机制做文本分类时的训练配置取舍", ""),
    ("其他信息服务_信息安全", "企业内网遭到钓鱼邮件后的应急处置", ""),
    ("学科教育_教育", "小学语文课堂的分组阅读教学设计", ""),
    ("文学_情感", "一个人回到旧居的那个傍晚", ""),
    ("水利_海洋", "汛期水库调度与近岸赤潮监测", ""),
    ("游戏", "开放世界游戏的关卡节奏设计", ""),
    ("科技_科学研究", "实验室里一次材料表征实验的记录", ""),
    ("采矿", "井下采掘工作面的支护与通风", ""),
    ("住宿_餐饮_酒店", "一家民宿的旺季运营与早餐安排", ""),
    ("其他制造", "车间里注塑件良率波动的排查", ""),
    ("影视_娱乐", "一部剧的剪辑节奏与配乐选择", ""),
    ("新闻传媒", "一次突发事件的现场报道流程", ""),
    ("汽车", "新能源车冬季续航衰减的原因", ""),
    ("生物医药", "单克隆抗体的纯化工序", ""),
    ("航空航天", "运载火箭二级发动机的试车", ""),
    ("金融_经济", "央行降准之后债券市场的反应", ""),
    ("体育", "一场篮球赛末节的战术调整", ""),
    ("农林牧渔", "水稻插秧期的水肥管理", ""),
    ("房地产_建筑", "一个住宅项目的施工节点与验收", ""),
    ("旅游_地理", "川西高原自驾路线的行程安排", ""),
    ("法律_司法", "一起买卖合同纠纷的判决理由", ""),
    ("电力能源", "光伏电站逆变器故障的排查", ""),
    ("计算机编程_代码", "用 asyncio 写一个带限流的爬虫", ""),
]

SYS = "你是一名中文写作者，习惯写具体的、有细节的连续文字。"
PROMPT = {
    # 工作记录口吻：第一人称、细节多、不带任何类目字眼
    "record": ("以「%s」为题写一段正文，450 到 600 字。"
               "要求：不分点、不加标题、不出现任何行业或学科的名字，"
               "就写成一段连续的自然文字，像从业者随手写的工作记录。"),
    # 行业报道口吻：更接近训练语料的样子，但同样不许出现类目名
    "article": ("以「%s」为题写一篇 450 到 600 字的行业报道。"
                "要求：有具体的事例、数字和引述，不分点、不加标题，"
                "不出现任何行业或学科的名字，写成连续的正文。"),
}
STYLE = "record"


def call(api_key, topic):
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PROMPT[STYLE] % topic},
        ],
        "temperature": 1.0,
        "max_tokens": 900,
        "stream": False,
    }, ensure_ascii=False)

    r = subprocess.run(
        ["curl.exe", "--ssl-no-revoke", "-s", "-S", "--max-time", "120",
         URL,
         "-H", "Content-Type: application/json",
         "-H", "Authorization: Bearer " + api_key,
         "--data-binary", "@-"],
        input=body.encode("utf-8"), capture_output=True, timeout=140,
    )
    raw = (r.stdout or b"").decode("utf-8", "replace")
    if not raw.strip():
        return None, "空响应 rc=%d %s" % (r.returncode, (r.stderr or b"")[:120])
    try:
        d = json.loads(raw)
    except Exception:
        return None, "非 JSON: " + raw[:120]
    if "error" in d:
        return None, str(d["error"])[:160]
    try:
        return d["choices"][0]["message"]["content"].strip(), None
    except Exception:
        return None, "结构异常: " + raw[:160]


def main():
    global STYLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--style", default="record", choices=["record", "article"])
    ap.add_argument("--repeat", type=int, default=1, help="每个题目生成几篇")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    STYLE = args.style
    out = args.out or (rpaths.sample("llm_articles_record.jsonl") if STYLE == "record"
                       else rpaths.sample("llm_articles_news.jsonl"))

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("环境变量 DEEPSEEK_API_KEY 没设置")

    todo = TOPICS[:args.limit] if args.limit else TOPICS
    todo = [t for t in todo for _ in range(max(1, args.repeat))]
    lock = threading.Lock()
    rows, errors = [], []

    def work(item):
        dom, topic, _ = item
        text, err = call(key, topic)
        with lock:
            if text and len(text) >= 120:
                rows.append({"domain": dom, "topic": topic, "text": " ".join(text.split())})
                print("OK   %-22s %5d 字  %s" % (dom, len(text), topic), flush=True)
            else:
                errors.append((dom, err))
                print("FAIL %-22s %s" % (dom, err), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    if rows:
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        avg = sum(len(r["text"]) for r in rows) / len(rows)
        print("\n成功 %d / %d，平均 %d 字，写入 %s" % (len(rows), len(todo), avg, out))
    else:
        print("\n一条都没成功，先看错误信息")


if __name__ == "__main__":
    main()
