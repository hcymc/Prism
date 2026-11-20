<p align="center">
<img alt="Prism logo" src="assets/logo/prism-logo.svg" width="160" height="160">
<h1 align="center">Prism-v3-exp</h1>
</p>



<p>
<img alt="version" src="https://img.shields.io/badge/version-v3-blue">
<img alt="status" src="https://img.shields.io/badge/status-experimental_preview-orange">
<img alt="model" src="https://img.shields.io/badge/model-prism__v3__exp-green">
<img alt="params" src="https://img.shields.io/badge/params-33.35M-blueviolet">
<img alt="license" src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-lightgrey">
</p>

**一个从零训练的中文小语言模型：一边猜下一个字，一边把整段文字拆成 31 个行业的概率光谱。**

一束白光穿过棱镜散成光谱，一段文字穿过 8 层散成 31 路概率。
本发布只包含一个模型：`prism-v3-exp`（129,600 步 / 30 类 / 33.35M 参数的最终权重）

**在线体验**：[model.hcy.ink](https://model.hcy.ink)　**权重下载**：[HuggingFace](https://huggingface.co/hcy-ai/prism-v3-exp) · [ModelScope](https://www.modelscope.cn/models/hcymcweb/Prism-v3-exp)

---

## 1. 这是什么

> **实验性质的中文领域分布模型**（experimental preview）
> 输入一段中文（≤512 字），输出一个 31 维的行业概率分布——不是给你一个标签，是给你一整条光谱。

- **一副身子两个出口**：8 层 decoder-only（d=512 · 8 头 · FFN 2048 · RoPE），
  语言头逐字猜下一个字，领域头在最后一个位置吐出 31 维分布；
- **从零训练**：手写 PyTorch，不依赖 `transformers` / `datasets`，从随机初始化开始，
  在 3.56 亿字中文行业语料上训 129,600 步；
- **四项损失**：语言 + 领域 + 两块一致性（对称 KL）+ 对比（InfoNCE），后两项不需要标签；
- **池化取最后一个位置**：decoder-only 里唯一看完全文的位置，前面都是「半瞎」的；
- **推理不需要 torch**：ONNX int8 只有 40 MB，2 核 2G 的机器单核 744 ms、常驻 136 MB。

**性质声明：这是实验预览，不是能直接用的打标工具。**
短于 300 字的文本命中率掉到 60% 以下；模糊文本（散文、杂谈、混合话题）它会硬选一个真类，
而且很有信心——因为第 31 类「其他」没有训练样本。详见 §6 与 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)。

## 1.5 效果示例

真实输出，没有挑样本。**同一套权重，一个判得准，一个判得离谱**——后者恰好说明了它的缺陷：

| 输入                                                                                 | 判定                          | 次选             |
| ------------------------------------------------------------------------------------ | ----------------------------- | ---------------- |
| 「央行今日下调中期借贷便利利率十个基点，市场普遍预期后续还有降准空间……」           | 金融_经济**96.7%**      | 房地产_建筑 0.6% |
| 「晨光刚漫过落地窗……菜市场里，粮油店的豆油换了标签……奶茶店排队，面馆的蒸汽……」 | 住宿_餐饮_酒店**95.4%** | 旅游_地理 0.3%   |

第二段是一篇写「城市生活一天」的散文，正确做法是「不确定」或者落到「其他」。
模型看到菜市场、奶茶、面馆、烤串就投了餐饮票——它抓到的是词，不是意思。

## 2. 快速开始

**不想装环境？先看效果：https://model.hcy.ink** —— 贴一段中文直接出分布，手机也能开。

环境：Python 3.9+、`torch>=2.0`、`numpy>=1.24`。
**只想推理的话不用装 torch**，`onnxruntime + numpy` 就够（约 156 MB）。

```
pip install -r requirements.txt
```

### 2.1 用 ONNX 推理（推荐，不装 torch）

仓库自带 `models/domain_int8.onnx`（40 MB）与词表：

```
pip install onnxruntime numpy
python src/serve_onnx.py --model models/domain_int8.onnx --meta models/model_meta.json
# → http://127.0.0.1:8902/  推理 / 模型卡 / 原理 三页
```

### 2.2 用完整 `.pt` 权重

完整权重（fp16，79 MB）放在模型托管平台，仓库里不放二进制：

| 平台 | 地址 |
| --- | --- |
| HuggingFace | https://huggingface.co/hcy-ai/prism-v3-exp |
| ModelScope | https://www.modelscope.cn/models/hcymcweb/Prism-v3-exp |

两个平台内容相同：`prism-v3-exp.pt`（79 MB）、`domain_fp32.onnx`（136 MB）、
`domain_int8.onnx`（40 MB）、`model_meta.json`、`SHA256SUMS`。

```
# 从 HF 下（国内建议加镜像）
hf download hcy-ai/prism-v3-exp --local-dir ./prism-weights
# 或 ModelScope
modelscope download --model hcymcweb/Prism-v3-exp --local_dir ./prism-weights
```

下好之后：

```
python src/infer.py --checkpoint ./prism-weights/prism-v3-exp.pt \
    --text "被告未按合同约定交付货物，一审法院判令其支付违约金及利息。" --topk 5
```

`.pt` 里带 `config / vocab / domains / step`，`torch.load` 之后直接构造模型，不需要额外参数文件：

```python
ck = torch.load("prism-v3-exp.pt", map_location="cpu")
model = TinyDomainLM(**ck["config"]).eval()
model.load_state_dict(ck["model"])       # fp16 会自动转成 fp32
```

### 2.3 网页推理界面

```
python src/webui.py --checkpoint /path/to/prism-v3-exp.pt --open
# torch 版，多一个「顺便续写」开关；ONNX 版没有语言头，这个开关不显示
```

### 2.4 训练监控面板

```
python tools/dashboard.py --run-dir data/runs/v3_30class --port 8901
# → http://127.0.0.1:8901/
```

零依赖（纯标准库 + 原生 canvas），读 `heartbeat.json` 判活、读 `train.log` 画曲线，
显示四项损失、验证命中率、熵、KL、困惑度、显存、步速，以及 best 出现在第几步。

## 3. 仓库内容

| 路径                        | 说明                                                           |
| --------------------------- | -------------------------------------------------------------- |
| `models/domain_int8.onnx` | **发布权重**（40 MB）。动态量化 int8，只含领域出口       |
| `models/model_meta.json`  | 词表 15,341 + 31 个类名 +`max_len` + 训练步数                |
| `src/`                    | 模型 / 训练 / 推理 / 两个网页入口 / 路径解析                   |
| `src/serve_onnx.py`       | 纯 ONNX 推理服务，只依赖 onnxruntime + numpy                   |
| `tools/`                  | 监控面板、开训体检、ONNX 导出与对账、性能基准                  |
| `eval/`                   | 13 个评测脚本，清单见[docs/EVAL.md](docs/EVAL.md)                 |
| `dataset/`                | 语料抓取（IndustryCorpus2），见[docs/DATASET.md](docs/DATASET.md) |
| `web/`                    | 三个页面的模板与样式（endfield 风格，含手机端适配）            |
| `docs/`                   | 模型卡、内部机制、评测、数据集、笔记                           |
| `scripts/`                | 启动脚本；`legacy/` 是两个已废弃的调度器                     |
| `assets/logo/`            | Logo（SVG 源 + 16–1024 px PNG + favicon），由 `tools/make_logo.py` 生成 |

所有路径经 `src/paths.py` 解析，代码里没有绝对路径。数据与产物默认在仓库内 `data/`（已忽略），
要放别处就设 `PRISM_HOME`。

## 4. 训练概况（详见 [MODEL_CARD](docs/MODEL_CARD.md) / [HOW_IT_WORKS](docs/HOW_IT_WORKS.md)）

| 项            | 值                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------- |
| 架构          | 8 层 decoder-only · d=512 · 8 头 · FFN 2048 · RoPE · Pre-LN                          |
| 参数          | 33,353,760（语言头与字表嵌入共享，领域头只占 0.83%）                                      |
| 词表 / 上下文 | 15,341 字符级 / 512 字                                                                    |
| 数据          | IndustryCorpus2，316,712 篇 / 356,481,811 字 / 30 个行业子集                              |
| 训练          | 129,600 步 · batch 8 · 块长 512（256–512 随机，长度增广）· lr 3e-4 余弦 · warmup 800 |
| 硬件          | 单卡 RTX 2080 SUPER 8G · fp16 · 峰值显存 2.99 GB · 约 6 小时                           |
| 效果          | 类均衡召回**85.2%**（600 篇）· 自然分布 88.45%（2000 篇）· 512 字输入 82–88%     |
| 量化          | int8 ONNX 与 fp32 同判率 99.7%，300 篇命中率不变                                          |

## 5. 全链路脚本（想从零重训时用）

```
src/prepare.py           语料目录 → blocks.pt + vocab.json + 软标签（按文档切分），--demo 造玩具数据
tools/preflight.py       开训前体检：两条出口 × 各种损失开关真跑 forward/backward，实测显存与步速
src/train.py             训练：四项损失、文档级双块、长度增广、按验证 KL 早停、心跳 + 快照
tools/dashboard.py       训练监控面板（零依赖）
src/infer.py             命令行推理
src/webui.py             torch 版网页（带续写）
src/serve_onnx.py        ONNX 版网页（不装 torch）
tools/export_onnx.py     导出 ONNX（fp32 / int8），顺带与 torch 对账
tools/export_weights.py  把训练存档剥成纯权重（fp16）
tools/bench_infer.py     单次延迟与内存
eval/*.py                13 个评测脚本
dataset/*.py             语料筛选、抓取、外部对照文本生成
```

三步跑通（玩具数据，不联网）：

```
python src/prepare.py --demo
python src/train.py --steps 200 --batch 8 --run-dir data/runs/smoke
python src/infer.py --checkpoint data/runs/smoke/ckpt_final.pt --text "原告与被告签订合同" --topk 3
```

## 6. 已知限制（完整清单见 [MODEL_CARD](docs/MODEL_CARD.md)）

- **短文本不可用**。30 字 44%，300 字 72%，512 字 82–88%。训练时永远见 512 字，短输入属于没见过的形态。
  重复补齐能救到 62.5%，但仍远低于长文。
- **不会说「不确定」**。31 个槽位里第 31 格「其他」一个训练样本都没有，模型学不会输出它。
  模糊文本只能从 30 个真类里硬选一个，而且常常很有信心（见 §1.5 第二个例子）。
- **类目重叠是硬天花板**。计算机_通信 50%、科技_科学研究 60%、电力能源 65%——
  不是分类头不会分，是那几堆向量本来就没分开，标签自己在打架。
- **输出不是真的软标签**。目标是「真类 0.95 + 其余 30 类均摊」，分布形状由标签平滑决定，
  不代表真实的领域共存比例。当权重用之前先做温度校准。
- **换一批文本掉 20 个点**。同类语料 85%，DeepSeek 写的 150 篇只有 64%——怀疑是只认训练集那种新闻体。
- **全角标点会压低置信度**（命中率影响在噪声内）。已由 `src/textproc.py` 做全角→半角归一化，
  自己调接口时别绕过它。

## 7. 作者与许可

作者：**hcy**

- 代码：**MIT**（见 `LICENSE`）
- 训练数据与权重：**Apache-2.0**（继承上游 `BAAI/IndustryCorpus2`）
- 界面字体 Saira：SIL Open Font License 1.1（`web/static/fonts/Saira-OFL.txt`）
- 界面规范取自 endfield-ui 套件，仅借结构与排版习惯，未使用任何游戏素材

本模型与 IndustryCorpus2 的发布方无隶属关系。

---

## 相关链接

| | |
| --- | --- |
| 在线体验（推理网页） | https://model.hcy.ink |
| GitHub（代码 / 训练脚本 / 网页 / 文档） | https://github.com/hcymc/Prism |
| HuggingFace（权重） | https://huggingface.co/hcy-ai/prism-v3-exp |
| ModelScope（权重） | https://www.modelscope.cn/models/hcymcweb/Prism-v3-exp |

