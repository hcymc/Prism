"""ONNX 版推理服务：只依赖 onnxruntime + numpy，不装 torch 也能跑。

给 2 核 2G 的小服务器用。模型导出成静态图（1×512），只保留领域出口，
不算 15,341 维的语言头——那一步占 15% 算力，也不需要。

    python3 serve_onnx.py --host 127.0.0.1 --port 8902 --model domain_int8.onnx

页面资源全用相对路径，所以挂在 nginx 的 /domain-lab/ 子路径下也能正常取到。
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np
import onnxruntime as ort

import paths
from textproc import encode as tp_encode

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL_DIR = os.path.join(PROJECT, "web", "templates")
STATIC_DIR = os.path.join(PROJECT, "web", "static")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

META = None
SESS = None
DOMAINS = None
VOCAB = None
INV = None
MAXLEN = 512
STEP = 0
LOCK = threading.Lock()

MAX_BODY = 256 * 1024       # 请求体上限，挡住拿大 body 打内存
MAX_TEXT = 20000            # 原文上限，超了先截，反正模型只看 512 字

# 落盘日志：给了 --log 就写文件（Windows 上靠这个留痕，systemd 那边用 journald 也行）
LOG_PATH = None
LOG_LOCK = threading.Lock()
LOG_MAX = 5 * 1024 * 1024


def log(level, msg):
    line = "%s [%-5s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), level, msg)
    if not LOG_PATH:
        print(line, flush=True)
        return
    with LOG_LOCK:
        try:
            if os.path.isfile(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX:
                os.replace(LOG_PATH, LOG_PATH + ".1")   # 只留一份旧的，不做无限滚动
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def predict(text, generate=0):
    text = (text or "").strip()[:MAX_TEXT]
    if not text:
        raise ValueError("文本为空")
    # 训练语料没有全角标点、也没见过 <unk>，先归一化再丢掉字表外字符
    ids, _dropped = tp_encode(text, VOCAB)
    ids = ids[:MAXLEN]
    if not ids:
        raise ValueError("没有可用字符")
    # 训练时永远见 512 字，短输入属于没见过的形态：把内容重复补齐
    if len(ids) < MAXLEN:
        ids = (ids * ((MAXLEN + len(ids) - 1) // len(ids)))[:MAXLEN]

    x = np.asarray([ids], dtype=np.int64)
    t0 = time.time()
    with LOCK:
        lg = SESS.run(None, {"ids": x})[0][0].astype(np.float64)
    lg = lg - lg.max()
    p = np.exp(lg)
    p /= p.sum()

    order = np.argsort(-p)
    dist = [{"name": DOMAINS[i], "p": round(float(p[i]), 5)} for i in order]
    ent = float(-(p * np.log(np.clip(p, 1e-9, None))).sum())
    return {
        "top": dist[:6],
        "dist": dist,
        "entropy": round(ent, 4),
        "n_chars": len(ids),
        "ms": round((time.time() - t0) * 1000, 1),
        "continuation": "",
        "model": {"step": STEP, "domains": len(DOMAINS), "device": "cpu · onnx",
                  "max_len": MAXLEN, "continuation": False},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DomainLab/1.0"

    def log_message(self, fmt, *args):
        # 只记页面和接口，静态资源不刷屏
        try:
            if '/static/' not in self.path:
                log("ACCESS", "%s %s %s" % (self.client_address[0], self.command, self.path))
        except Exception:
            pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200, cache="no-store"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _page(self, name):
        path = os.path.join(TPL_DIR, name)
        if not os.path.isfile(path):
            self.send_error(404, "template missing")
            return
        with open(path, "r", encoding="utf-8") as f:
            self._send(f.read(), "text/html; charset=utf-8")

    def _static(self, rel):
        rel = unquote(rel.split("?", 1)[0])
        if ".." in rel or rel.startswith("/"):
            self.send_error(403)
            return
        path = os.path.join(STATIC_DIR, rel)
        ext = os.path.splitext(path)[1].lower()
        if ext not in MIME or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            self._send(f.read(), MIME[ext], cache="public, max-age=86400")

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index", "/index.html"):
            self._page("index.html")
        elif p in ("/card", "/card/"):
            self._page("model_card.html")
        elif p in ("/how", "/how/"):
            self._page("how.html")
        elif p.startswith("/static/"):
            self._static(p[len("/static/"):])
        elif p == "/api/state":
            self._send(json.dumps({"model": {
                "step": STEP, "domains": len(DOMAINS), "device": "cpu · onnx",
                "max_len": MAXLEN, "continuation": False}}, ensure_ascii=False))
        elif p == "/healthz":
            self._send("ok", "text/plain; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > MAX_BODY:
                self._send(json.dumps({"error": "请求体过大"}, ensure_ascii=False), code=413)
                return
            body = json.loads(self.rfile.read(n) or b"{}")
            # 续写这版不开：40 个字 = 40 次完整前向，1 核上要十几秒
            res = predict(body.get("text", ""))
            self._send(json.dumps(res, ensure_ascii=False))
            top = res["top"][0]
            log("PRED", "%s %d 字 -> %s %.1f%% %sms"
                % (self.client_address[0], res["n_chars"], top["name"],
                   top["p"] * 100, res["ms"]))
        except Exception as e:
            self._send(json.dumps({"error": str(e)[:200]}, ensure_ascii=False), code=400)
            log("ERROR", "%s %s: %s" % (self.client_address[0], type(e).__name__, str(e)[:160]))


def main():
    global META, SESS, DOMAINS, VOCAB, INV, MAXLEN, STEP, LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--model", default=paths.onnx(), help="ONNX 权重路径")
    ap.add_argument("--meta", default=paths.onnx("model_meta.json"), help="词表与类别名")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime 线程数，0=按核数")
    ap.add_argument("--log", default="", help="日志文件路径，给了就写盘，超过 5MB 轮转一份")
    args = ap.parse_args()

    if args.log:
        LOG_PATH = os.path.abspath(args.log)
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        if not os.path.exists(LOG_PATH):
            # 带 BOM：PowerShell 5.1 的 Get-Content 和记事本靠它认出是 UTF-8，否则中文按 ANSI 读成乱码
            with open(LOG_PATH, "wb") as f:
                f.write(b"\xef\xbb\xbf")

    with open(args.meta, encoding="utf-8") as f:
        META = json.load(f)
    VOCAB = META["vocab"]
    INV = {i: c for c, i in VOCAB.items()}
    DOMAINS = META["domains"]
    MAXLEN = META["max_len"]
    STEP = META["step"]

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    SESS = ort.InferenceSession(args.model, so,
                                providers=["CPUExecutionProvider"])

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log("BOOT", "监听 http://%s:%d  模型 %s  线程 %d/%d  pid=%d  py=%s"
        % (args.host, args.port, args.model, so.intra_op_num_threads,
           os.cpu_count(), os.getpid(), ".".join(map(str, sys.version_info[:3]))))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("BOOT", "收到中断，退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
