# 权重

## 仓库里的（可以直接跑）

| 文件 | 体积 | 说明 |
| --- | --- | --- |
| `domain_int8.onnx` | 40 MB | **发布权重**。动态量化 int8，只含领域出口 |
| `model_meta.json` | 200 KB | 词表 15,341 + 31 个类名 + `max_len` + 训练步数 |

```bash
pip install onnxruntime numpy
python src/serve_onnx.py --model models/domain_int8.onnx --meta models/model_meta.json
```

不需要 torch。与 fp32 的同判率 99.7%（300 篇里 1 篇不同），命中率不变。

## 放在模型托管平台的（体积大）

| 平台 | 地址 |
| --- | --- |
| HuggingFace | https://huggingface.co/hcy-ai/prism-v3-exp |
| ModelScope | https://www.modelscope.cn/models/hcymcweb/Prism-v3-exp |

| 文件 | 体积 | 说明 |
| --- | --- | --- |
| `prism-v3-exp.pt` | 79 MB | 完整权重（fp16），带 `config / vocab / domains / step` |
| `domain_fp32.onnx` | 136 MB | 未量化的 ONNX，与 torch 输出逐位相同 |
| `domain_int8.onnx` | 40 MB | 与仓库里那份相同，方便一次性下全 |
| `model_meta.json` | 200 KB | 词表 + 类名 |
| `SHA256SUMS` | — | 校验和，下完自己对一遍 |

```bash
hf download hcy-ai/prism-v3-exp --local-dir ./prism-weights
# 或
modelscope download --model hcymcweb/Prism-v3-exp --local_dir ./prism-weights
```

下载 `.pt` 之后：

```python
import torch
from model import TinyDomainLM            # src/ 下

ck = torch.load("prism-v3-exp.pt", map_location="cpu")
model = TinyDomainLM(**ck["config"]).eval()
model.load_state_dict(ck["model"])        # fp16 载入 fp32 模型会自动转换
```

或者直接给命令行脚本：

```bash
python src/infer.py --checkpoint /path/to/prism-v3-exp.pt --text "……" --topk 5
python src/webui.py --checkpoint /path/to/prism-v3-exp.pt --open
```

## 为什么 `.pt` 不放进仓库

原始训练存档 `best.pt` 是 382 MB——里面 2/3 是 AdamW 的一阶、二阶矩和随机数状态，发布用不上。
剥成纯权重再存 fp16 之后是 79 MB，仍然不适合进 git（每次克隆都要拉一遍二进制）。

所以：**仓库只带 40 MB 的 int8 ONNX，完整权重走模型托管平台。**

## 自己导出

```bash
# 训练存档 → 纯权重（fp16）
python tools/export_weights.py --checkpoint runs/v3_30class/best.pt --out models/prism-v3-exp.pt

# 训练存档 → ONNX（fp32 + int8），顺带与 torch 对账
python tools/export_onnx.py --out data/onnx
python tools/eval_onnx.py --docs 300        # 预期 int8 与 torch 同判 99%+
```

导出 ONNX 时做了两件事：只导领域出口（语言头占 15% 算力，推理用不到）；
把 `scaled_dot_product_attention` 展开成显式 matmul + mask + softmax（torch 2.2 的 SDPA 导不出来）。

## 一个容易踩的点

`prism-v3-exp.pt` 的 `state_dict` 里 `tok.weight` 和 `lm_head.weight` 是同一份张量（权重共享），
所以把张量元素个数加起来会得到 41,208,352，比真实参数量 33,353,760 多出 7,854,592。
这不是存重了，是 `state_dict` 按名字列了两遍。加载时两者都要在，`strict=True` 才通得过。
