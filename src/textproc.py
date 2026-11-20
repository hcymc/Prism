"""把真实世界的中文标点，还原成训练语料里的样子。

训练语料（IndustryCorpus2）里一个全角标点都没有，逗号句号全是半角的 `,` `.`，
引号是 `"`。所以字表 15341 个字里，`，。、《》「」` 一个都不存在。

推理时如果原样喂进去，每个标点都会变成 <unk>——而 <unk> 在训练数据里出现 0 次，
它的向量是随机初始化的。等于每十个字插一个随机噪声，句子边界也全断了。

这里做两件事：
  1. 全角 → 半角，和训练语料一致；
  2. 字表里仍然查不到的字直接丢掉，不塞 <unk>（训练时没有这个符号，塞进去只是噪声）。
"""

# 全角 → 半角。左边是真实文本里会出现的，右边是训练语料里的样子。
FULL2HALF = {
    "，": ",", "。": ".", "、": ",", "；": ";", "：": ":",
    "？": "?", "！": "!", "“": '"', "”": '"', "‘": "'", "’": "'",
    "（": "(", "）": ")", "《": "<", "》": ">", "【": "[", "】": "]",
    "「": '"', "」": '"', "『": '"', "』": '"',
    "—": "-", "–": "-", "―": "-", "－": "-",
    "…": ".", "·": ".", "～": "~", "　": " ",
    "％": "%", "＋": "+", "＝": "=", "／": "/", "＼": "\\",
    "＆": "&", "＊": "*", "＃": "#", "＠": "@",
}
# 全角数字和字母
for _i in range(10):
    FULL2HALF[chr(0xFF10 + _i)] = chr(ord("0") + _i)
for _i in range(26):
    FULL2HALF[chr(0xFF21 + _i)] = chr(ord("A") + _i)
    FULL2HALF[chr(0xFF41 + _i)] = chr(ord("a") + _i)

TABLE = str.maketrans(FULL2HALF)


def to_training_form(text):
    """全角标点换成半角，其余原样。"""
    return (text or "").translate(TABLE)


def encode(text, vocab, drop_unknown=True):
    """字符 → id。先归一化，再逐字查表；查不到的丢掉（而不是塞 <unk>）。

    返回 (ids, 丢弃字数)。
    """
    unk = vocab["<unk>"]
    ids, dropped = [], 0
    for ch in to_training_form(text):
        if ch.isspace():
            continue
        i = vocab.get(ch)
        if i is None:
            dropped += 1
            if not drop_unknown:
                ids.append(unk)
            continue
        ids.append(i)
    return ids, dropped
