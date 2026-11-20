"""全项目的路径解析，只此一处。

代码里不出现任何绝对路径。数据、权重、训练产物默认放在 <repo>/runs 下，
想放到别处（比如另一块盘）就设环境变量：

    Windows:     set PRISM_HOME=D:/prism-data
    bash:        export PRISM_HOME=~/prism-data

目录约定（PRISM_HOME 之下）：

    blocks.pt                prepare.py 切好的训练块
    vocab.json               词表
    runs/<run>/{best,latest}.pt
    runs/<run>/train.log     训练日志
    onnx/                    导出的 ONNX
    samples/                 手写样例、外部模型生成的测试文本

仓库自带的那份配置见 README 的「准备数据」一节。
"""

import os

# 仓库根目录（本文件在 src/ 下）
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据与产物的老家。没设环境变量就用仓库里的 data/（已加进 .gitignore）
HOME = os.path.abspath(os.environ.get("PRISM_HOME") or os.path.join(PROJECT, "data"))

DEFAULT_RUN = "v3_30class"


def path(*parts):
    """PRISM_HOME 下的某个路径。"""
    return os.path.join(HOME, *parts)


def blocks():
    """prepare.py 的产物。"""
    return path("blocks.pt")


def vocab():
    return path("vocab.json")


def run_dir(run=None):
    return path("runs", run or DEFAULT_RUN)


def ckpt(run=None, name="best.pt"):
    """某个 run 的权重。"""
    return os.path.join(run_dir(run), name)


def onnx(name="domain_int8.onnx"):
    return path("onnx", name)


def sample(name):
    return path("samples", name)


def ensure_home():
    """第一次运行时把目录建出来，省得各脚本各自 mkdir。"""
    for sub in ("", "runs", "onnx", "samples"):
        os.makedirs(os.path.join(HOME, sub), exist_ok=True)
    return HOME


def describe():
    """给命令行脚本打印一行“我在哪找东西”，出问题好排查。"""
    return "PRISM_HOME = %s%s" % (HOME, "" if os.path.isdir(HOME) else "  (目录不存在)")
