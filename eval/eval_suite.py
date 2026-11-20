"""对当前最好的 checkpoint 做「短 / 中 / 长」三档测试，三个来源。

来源 1  测试集：按文档切出来的验证集（训练时从未见过），有标准答案，可算命中率
来源 2  网上文章：维基百科中文正文（经 Clash 代理抓取）+ 中国新闻网频道稿
来源 3  自写虚拟文本：6 个领域 x 3 种长度，人写的，不带语料库的文风

GPU 正被训练占着（只剩 700MB），这里强制跑 CPU、限 2 线程，不抢训练的算力。
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


# ---------------------------------------------------------------- 自写虚拟文本
# 每条都写明「期望领域」，长度按 30 / 120 / 400 字三档写
VIRTUAL = [
    # 金融_经济
    {"domain": "金融_经济", "len": "短",
     "text": "央行今日宣布降准0.25个百分点，释放长期资金约五千亿元。"},
    {"domain": "金融_经济", "len": "中",
     "text": "央行今日宣布下调金融机构存款准备金率0.25个百分点，释放长期资金约五千亿元。"
             "分析人士认为，此次降准主要用于对冲季末流动性紧张，同时配合财政发力稳增长，"
             "市场普遍预期后续贷款市场报价利率仍有小幅下行空间。"},
    {"domain": "金融_经济", "len": "长",
     "text": "本周银行间市场资金面先紧后松。周初受缴税与地方债发行缴款双重影响，隔夜回购利率一度冲高至2.3%附近，"
             "非银机构融资难度明显加大，部分券商资管被迫在尾盘折价融入。周三央行开展中期借贷便利操作，净投放三千亿元，"
             "并重启十四天期逆回购，资金价格随即回落。同业存单发行方面，国有大行一年期品种利率小幅下行三个基点，"
             "股份行与城商行分化加剧，评级较低的机构认购倍数明显不足。债券市场方面，十年期国债收益率全周下行两个基点，"
             "信用利差继续走阔，地产债成交清淡，机构普遍采取缩短久期的防御策略。展望后市，多数分析师认为资金面仍将维持紧平衡，"
             "但进一步收紧的概率不高，建议关注跨月时点的流动性扰动与财政支出节奏。"},
    # 体育
    {"domain": "体育", "len": "短",
     "text": "昨晚中超联赛，主队补时阶段头球破门，二比一逆转客队。"},
    {"domain": "体育", "len": "中",
     "text": "昨晚进行的中超联赛第28轮，主队在先失一球的情况下连扳两球，最终二比一逆转客队。"
             "下半场第87分钟，替补登场的年轻边锋右路传中，中锋抢点头球破门完成绝杀，"
             "全场球迷起立鼓掌。这场胜利让主队暂时升至积分榜第三位。"},
    {"domain": "体育", "len": "长",
     "text": "比赛开始阶段客队踢得更有侵略性，前场高位逼抢迫使主队多次出现后场传球失误。第23分钟，客队中路直塞撕开防线，"
             "前锋单刀推射远角得手，取得一比零领先。丢球之后主队逐渐加强中场控制，通过两个边路的轮转拉开宽度，"
             "但上半场始终未能形成有效射门，半场技术统计显示主队控球率超过六成，射正次数却是零。易边再战，"
             "主教练换上两名速度型球员改打四四二，边路传中次数明显增加。第68分钟主队角球战术奏效，中卫前点头球后蹭扳平比分。"
             "此后比赛进入拉锯，双方体能下降明显，犯规次数增多。第87分钟，替补登场的年轻边锋右路突破后传中，"
             "中锋在小禁区线上抢点头球破门完成绝杀。赛后新闻发布会上，主教练表示球队上半场过于保守，"
             "换人是比赛的转折点，同时强调保级形势依然严峻，不能因为一场胜利而松懈。"},
    # 法律_司法
    {"domain": "法律_司法", "len": "短",
     "text": "法院一审判决被告支付违约金十二万元，并承担全部诉讼费用。"},
    {"domain": "法律_司法", "len": "中",
     "text": "法院经审理认为，双方签订的商品房买卖合同合法有效，被告未按约定期限交付房屋已构成违约。"
             "依照民法典第五百七十七条、第五百八十五条之规定，判决被告支付违约金十二万元，并承担本案全部诉讼费用。"
             "如不服本判决，可在判决书送达之日起十五日内提起上诉。"},
    {"domain": "法律_司法", "len": "长",
     "text": "原告与被告于二〇二二年三月签订商品房买卖合同，约定被告应于同年十二月三十一日前交付房屋，"
             "逾期超过九十日的，买受人有权解除合同并要求支付总价款百分之五的违约金。合同签订后，原告依约支付全部购房款，"
             "被告因资金链问题迟迟未能竣工验收，实际交付时间迟至二〇二四年六月。庭审中，被告辩称延期交房系因疫情管控"
             "导致施工停滞，属于不可抗力，应当免除违约责任，并提交了当地住建部门出具的停工通知作为证据。"
             "法院经审理认为，疫情管控措施在一定期间内确实构成不可抗力，但被告提交的证据仅能证明停工六十二日，"
             "其余延期期间系其自身资金安排不当所致，不属于免责事由。关于违约金标准，被告主张约定比例过分高于实际损失，"
             "请求予以调减。法院结合同地段同类房屋租金水平、原告实际损失及双方过错程度，认为约定标准并未显失公平，"
             "不予调减。最终依照民法典第五百七十七条、第五百八十五条、第五百九十条之规定，判决被告支付违约金十二万元，"
             "并承担本案全部诉讼费用。如不服本判决，可在判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本。"},
    # 医学
    {"domain": "医学_健康_心理_中医", "len": "短",
     "text": "患者术后第三天出现低热，白细胞计数升高，考虑切口感染。"},
    {"domain": "医学_健康_心理_中医", "len": "中",
     "text": "患者男性，五十六岁，结肠癌根治术后第三天出现低热，体温波动在37.8至38.4摄氏度之间，"
             "切口周围皮肤红肿、压痛明显，血常规示白细胞12.6乘十的九次方每升，中性粒细胞比例偏高，"
             "C反应蛋白明显升高，考虑切口感染，已留取分泌物培养并经验性使用头孢类抗生素。"},
    {"domain": "医学_健康_心理_中医", "len": "长",
     "text": "患者男性，五十六岁，因升结肠腺癌行右半结肠根治术，手术过程顺利，术中出血约一百毫升，未输血。"
             "术后第一天生命体征平稳，肠鸣音弱，予禁食、补液、预防性抗生素治疗。术后第三天晨起出现低热，"
             "体温波动在37.8至38.4摄氏度之间，伴切口周围皮肤红肿、皮温升高、压痛明显，无明显反跳痛。"
             "实验室检查示白细胞12.6乘十的九次方每升，中性粒细胞比例百分之八十六，C反应蛋白升至六十八毫克每升，"
             "降钙素原轻度升高。腹部超声未见明显腹腔积液及脓肿形成。结合病史与体征，考虑浅表切口感染可能性大，"
             "已拆除部分缝线充分引流，留取分泌物行细菌培养及药敏试验，经验性予第二代头孢菌素静脉滴注。"
             "同时加强切口换药，每日一次，注意观察引流液性状与量。需警惕感染向深部蔓延，若出现持续高热、"
             "腹痛加剧或白细胞进行性升高，应及时复查腹部增强CT，排除吻合口漏与腹腔脓肿。"
             "营养支持方面，待肠功能恢复后尽早恢复肠内营养，有助于切口愈合。"},
    # 计算机编程_代码
    {"domain": "计算机编程_代码", "len": "短",
     "text": "用 asyncio.gather 并发请求时，记得加信号量限制并发数。"},
    {"domain": "计算机编程_代码", "len": "中",
     "text": "用 asyncio 批量请求下游接口时，直接 gather 上千个协程会把连接池打爆，报 too many open files。"
             "正确做法是套一层 asyncio.Semaphore(50) 限制并发，并用 asyncio.as_completed 边完成边处理，"
             "避免结果全部堆在内存里；超时要单独设，别指望外层 wait_for 兜底。"},
    {"domain": "计算机编程_代码", "len": "长",
     "text": "这段脚本的目标是从接口拉取十万条记录并写进数据库。最初直接用 asyncio.gather 一次性创建所有协程，"
             "跑起来立刻报 too many open files，原因是连接池上限只有一百，而协程数量是它的十倍以上。"
             "改法有三个层次：第一，用 asyncio.Semaphore 把并发压到五十以内，超出的协程会在 acquire 处排队，"
             "不会真正建立连接；第二，把 gather 换成 as_completed，边完成边消费，避免十万条结果同时驻留内存；"
             "第三，给每个请求单独套 asyncio.wait_for 设三秒超时，不要依赖外层整体超时，否则一个慢请求会拖住整批。"
             "写入侧同理，用批量插入代替逐条 commit，每五百条提交一次，配合 executemany 可以把写入耗时压到原来的十分之一。"
             "另外要注意异常处理，gather 默认遇到第一个异常就会向上抛，其余协程不会被取消，会造成连接泄漏，"
             "建议加 return_exceptions=True 再逐个检查结果。最后，记得在 finally 里显式关闭 session，"
             "否则事件循环退出时会有大量未关闭连接的警告。"},
    # 旅游_地理
    {"domain": "旅游_地理", "len": "短",
     "text": "十月的雨崩村，徒步进山要四个小时，沿途全是原始森林。"},
    {"domain": "旅游_地理", "len": "中",
     "text": "十月的雨崩村已经进入最佳季节，从西当村徒步进山大约需要四到五个小时，沿途穿过成片的原始冷杉林，"
             "翻过海拔三千七百米的那宗垭口后，梅里雪山的卡瓦格博峰突然出现在眼前。村里住宿以家庭客栈为主，"
             "旺季最好提前预订，晚上没有路灯，记得带头灯。"},
    {"domain": "旅游_地理", "len": "长",
     "text": "雨崩村位于云南迪庆藏族自治州德钦县，坐落在梅里雪山脚下，是澜沧江支流旁一个只有三十多户人家的小村落，"
             "至今不通公路，进出只能徒步或骑骡子。从西当村出发的进山路线全长约十二公里，累计爬升一千一百米，"
             "正常体力需要四到五个小时，沿途依次经过热水塘、八一茶馆和那宗垭口，翻过海拔三千七百米的垭口之后，"
             "卡瓦格博峰会在右手边突然展开，那是整条路线最值得停留的位置。村里分上雨崩和下雨崩两部分，"
             "上雨崩看神女峰角度最好，下雨崩离神瀑更近，住宿以家庭客栈为主，标间价格在旺季会翻倍，"
             "最好提前一周预订。十月是最佳季节，雨水少、能见度高，但早晚温差接近二十度，需要带抓绒和冲锋衣。"
             "村内没有路灯，夜间出行务必带头灯；手机信号只有移动勉强可用，电信基本无服务。"
             "如果时间充裕，可以安排一天去冰湖，往返约八小时，中途需要跨过一段碎石坡，雨天路滑要格外小心。"},
]

LENGTHS = [30, 80, 150, 300, 512]
BUCKET = {30: "短", 80: "短", 150: "中", 300: "长", 512: "长"}


def encode(text, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(c, unk) for c in text if not c.isspace()]


@torch.no_grad()
def predict(model, seqs, batch=12):
    """seqs: list[list[int]]，长度必须一致（模型用最后一个位置池化，右侧补 pad 会污染）。"""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        x = torch.tensor(chunk, dtype=torch.long)
        logits = model(x, lm_weight=0.0)["domain_logits"]
        p = torch.softmax(logits.float(), dim=-1)
        out.extend(p.tolist())
    return out


def top_of(probs, domains, k=3):
    order = sorted(range(len(probs)), key=lambda i: -probs[i])[:k]
    return [(domains[i], probs[i]) for i in order]


def ent_of(probs):
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def stats_block(name, rows):
    """rows: [(probs, true_domain_or_None)]"""
    n = len(rows)
    ok = sum(1 for p, t in rows if t and max(range(len(p)), key=lambda i: p[i]) == t)
    conf = sum(max(p) for p, _ in rows) / max(n, 1)
    ent = sum(ent_of(p) for p, _ in rows) / max(n, 1)
    return {"name": name, "n": n, "acc": ok / max(n, 1), "conf": conf, "ent": ent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--test-docs", type=int, default=90, help="测试集取多少篇（每类均匀）")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json-out", default=paths.path("runs/v3_30class/eval_suite.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")  # 显存留给训练，这里只用 CPU

    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    domains, vocab = ck["domains"], ck["vocab"]
    dom_id = {d: i for i, d in enumerate(domains)}
    maxlen = ck["config"]["max_len"]

    blob = torch.load(args.blocks, map_location="cpu")
    ids_all, off, dom_all, split = blob["ids"], blob["offsets"], blob["domain"], blob["split"]

    print("checkpoint = %s" % args.checkpoint)
    print("训练步数 = %s | 参数 %.2f 万 | 领域 %d 类 | 设备 %s / %d 线程"
          % (ck.get("step"), sum(p.numel() for p in model.parameters()) / 1e4,
             len(domains), device, args.threads))

    result = {"checkpoint": args.checkpoint, "step": ck.get("step"), "sections": {}}
    t_all = time.time()

    # ------------------------------------------------ 来源 1：测试集（有标准答案）
    print("\n=== 来源 1：测试集（按文档切分，训练未见过） ===")
    per_class = defaultdict(list)
    for d in (split == 1).nonzero().flatten().tolist():
        if int(off[d + 1]) - int(off[d]) >= maxlen:
            per_class[int(dom_all[d])].append(d)
    take = max(1, args.test_docs // max(len(per_class), 1))
    docs = []
    for cls in sorted(per_class):
        docs.extend(per_class[cls][:take])
    print("取 %d 篇（%d 类，每类 %d 篇，要求原文 >= %d 字）" % (len(docs), len(per_class), take, maxlen))

    test_rows = {}
    for L in LENGTHS:
        t0 = time.time()
        seqs, trues = [], []
        for d in docs:
            s = int(off[d])
            seqs.append(ids_all[s:s + L].long().tolist())
            trues.append(int(dom_all[d]))
        probs = predict(model, seqs)
        rows = list(zip(probs, trues))
        st = stats_block("test@%d" % L, rows)
        test_rows[L] = st
        print("  %3d 字  n=%3d  命中 %5.1f%%  置信 %.3f  熵 %.3f  (%.0fs)"
              % (L, st["n"], st["acc"] * 100, st["conf"], st["ent"], time.time() - t0))
    result["sections"]["test_set"] = {str(k): v for k, v in test_rows.items()}

    # ------------------------------------------------ 来源 2：网上文章
    web_sets = []
    for path, tag in [(paths.sample("web_wiki.json"), "维基百科"),
                      (paths.sample("web_samples.json"), "中国新闻网")]:
        if os.path.isfile(path):
            try:
                doc = json.load(open(path, encoding="utf-8"))
                if doc.get("items"):
                    web_sets.append((tag, doc["items"]))
            except Exception as e:
                print("读 %s 失败：%s" % (path, e))

    web_rows = {}
    for tag, items in web_sets:
        print("\n=== 来源 2：网上文章 · %s（%d 篇） ===" % (tag, len(items)))
        per_len = {}
        for L in LENGTHS:
            t0 = time.time()
            seqs, trues = [], []
            for it in items:
                eid = encode(it["text"], vocab)[:L]
                if len(eid) < 8:
                    continue
                seqs.append(eid)
                trues.append(dom_id.get(it["domain"]))
            probs = predict(model, seqs)
            rows = list(zip(probs, trues))
            st = stats_block("%s@%d" % (tag, L), rows)
            per_len[L] = st
            web_rows[(tag, L)] = (rows, items)
            print("  %3d 字  n=%3d  命中 %5.1f%%  置信 %.3f  熵 %.3f  (%.0fs)"
                  % (L, st["n"], st["acc"] * 100, st["conf"], st["ent"], time.time() - t0))
        result["sections"].setdefault("web", {})[tag] = {str(k): v for k, v in per_len.items()}

    # 逐篇明细（取 512 字那档）
    for (tag, L), (rows, items) in web_rows.items():
        if L != maxlen:
            continue
        print("\n  %s 逐篇明细（%d 字输入）：" % (tag, L))
        n = 0
        for (probs, true), it in zip(rows, items):
            if true is None:
                continue
            top = top_of(probs, domains, 3)
            flag = "OK " if top[0][0] == domains[true] else "ERR"
            print("   %s %-14s → %-14s %5.1f%%  | 次 %-12s %4.1f%%" %
                  (flag, domains[true], top[0][0], top[0][1] * 100, top[1][0], top[1][1] * 100))
            n += 1
        print("   （共 %d 篇）" % n)

    # ------------------------------------------------ 来源 3：自写虚拟文本
    print("\n=== 来源 3：自写虚拟文本（%d 条） ===" % len(VIRTUAL))
    vrows = defaultdict(list)
    detail = []
    for it in VIRTUAL:
        eid = encode(it["text"], vocab)[:maxlen]
        probs = predict(model, [eid])[0]
        true = dom_id[it["domain"]]
        top = top_of(probs, domains, 3)
        ok = top[0][0] == it["domain"]
        vrows[it["len"]].append((probs, true))
        detail.append({"len": it["len"], "domain": it["domain"], "chars": len(it["text"]),
                       "pred": top[0][0], "p": round(top[0][1], 4), "ok": ok,
                       "top3": [(d, round(p, 4)) for d, p in top]})
        print("  %s %-2s %3d字  %-14s → %-14s %5.1f%%  | 次 %-12s %4.1f%%" %
              ("OK " if ok else "ERR", it["len"], len(it["text"]), it["domain"],
               top[0][0], top[0][1] * 100, top[1][0], top[1][1] * 100))
    for k in ("短", "中", "长"):
        r = vrows.get(k) or []
        if r:
            st = stats_block("virtual@%s" % k, r)
            print("  汇总 %-2s  n=%2d  命中 %5.1f%%  置信 %.3f" % (k, st["n"], st["acc"] * 100, st["conf"]))
            result["sections"].setdefault("virtual", {})[k] = st
    result["sections"]["virtual_detail"] = detail

    # ------------------------------------------------ 三档总表
    print("\n=== 按长度三档汇总（测试集，样本量最一致） ===")
    for k in ("短", "中", "长"):
        lens = [L for L in LENGTHS if BUCKET[L] == k]
        if not lens:
            continue
        acc = sum(test_rows[L]["acc"] * test_rows[L]["n"] for L in lens) / \
              sum(test_rows[L]["n"] for L in lens)
        print("  %-2s（%s 字）  命中 %5.1f%%" % (k, "/".join(str(L) for L in lens), acc * 100))

    print("\n总耗时 %.1f 分钟" % ((time.time() - t_all) / 60))
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    print("结果写入 %s" % args.json_out)


if __name__ == "__main__":
    main()
