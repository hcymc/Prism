"""领域分布推理 WebUI：贴一段文字，看它在 31 个领域槽位上各占几成。

纯标准库 HTTP + 原生前端，不引 CDN。页面模板在 templates/，样式与字体在
static/（endfield-ui 套件 + 本地变体）。模型启动时加载一次常驻，之后每次
请求只做一次前向（512 字以内，几十毫秒）。

  GET  /              推理页面
  GET  /card          模型卡
  GET  /static/...    样式、字体、等高线
  GET  /api/state     模型信息
  POST /api/predict   {"text": "...", "generate": 0|N}
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from model import TinyDomainLM
from textproc import encode as tp_encode
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "..", "web")
TPL_DIR = os.path.join(WEB_DIR, "templates")
STATIC_DIR = os.path.join(WEB_DIR, "static")

CK = MODEL = DEV = None
VOCAB = DOMAINS = INV = None
MAXLEN = 512

LOCK = threading.Lock()

MIME = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".txt": "text/plain; charset=utf-8",
}


def load(path, want="auto"):
    global CK, MODEL, DEV, VOCAB, DOMAINS, INV, MAXLEN
    CK = torch.load(path, map_location="cpu")
    DEV = torch.device("cpu")
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        # 训练在跑时显存很紧，先看剩余量再决定要不要上卡
        try:
            free, _total = torch.cuda.mem_get_info()
            if want == "cuda" or free > 1400 * 1024 * 1024:
                DEV = torch.device("cuda")
        except Exception:
            DEV = torch.device("cpu")
    MODEL = TinyDomainLM(**CK["config"]).to(DEV)
    MODEL.load_state_dict(CK["model"])
    MODEL.eval()
    VOCAB, DOMAINS = CK["vocab"], CK["domains"]
    INV = {v: k for k, v in VOCAB.items()}
    MAXLEN = CK["config"]["max_len"]


def predict(text, generate=0):
    text = (text or "").strip()
    if not text:
        raise ValueError("文本为空")
    # 训练语料没有全角标点、也没见过 <unk>，所以先归一化再丢掉查不到的字
    ids, _dropped = tp_encode(text, VOCAB)
    ids = ids[:MAXLEN]
    if not ids:
        raise ValueError("没有可用字符")    # 训练时永远见 512 字，短输入会掉进没见过的形态：把内容重复补齐到 512
    # 实测同一批文档，前 30 字直接判命中 29.2%，重复补齐后 62.5%
    if len(ids) < MAXLEN:
        ids = (ids * ((MAXLEN + len(ids) - 1) // len(ids)))[:MAXLEN]
    x = torch.tensor([ids], dtype=torch.long, device=DEV)
    t0 = time.time()
    with LOCK, torch.no_grad():
        out = MODEL(x, lm_weight=0.0)
        p = torch.softmax(out["domain_logits"].float(), dim=-1)[0].cpu()
        cont = ""
        if generate > 0:
            gen = MODEL.generate(x, max_new=int(generate))
            cont = "".join(INV.get(int(i), "") for i in gen[0]
                           if INV.get(int(i), "") not in {"<pad>", "<unk>", "<bos>"}).replace("<eos>", " / ")
    order = sorted(range(len(DOMAINS)), key=lambda i: -p[i].item())
    dist = [{"name": DOMAINS[i], "p": round(p[i].item(), 5)} for i in order]
    ent = float(-(p * p.clamp(min=1e-9).log()).sum())
    return {
        "top": dist[:6],
        "dist": dist,
        "entropy": round(ent, 4),
        "n_chars": len(ids),
        "ms": round((time.time() - t0) * 1000, 1),
        "continuation": cont,
        "model": {"step": CK.get("step"), "domains": len(DOMAINS), "device": str(DEV)},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200, cache="no-store"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
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
        rel = rel.split("?", 1)[0]
        if ".." in rel or rel.startswith("/"):
            self.send_error(403)
            return
        path = os.path.join(STATIC_DIR, rel)
        ext = os.path.splitext(path)[1].lower()
        if ext not in MIME or not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            self._send(f.read(), MIME[ext], cache="public, max-age=3600")

    def do_GET(self):
        p = self.path.split("?", 1)[0].split("#", 1)[0]
        if p == "/" or p.startswith("/index"):
            self._page("index.html")
        elif p.startswith("/card"):
            self._page("model_card.html")
        elif p.startswith("/how"):
            self._page("how.html")
        elif p.startswith("/static/"):
            self._static(p[len("/static/"):])
        elif p.startswith("/api/state"):
            self._send(json.dumps({
                "model": {"step": CK.get("step"), "domains": len(DOMAINS), "device": str(DEV),
                          "max_len": MAXLEN, "continuation": True},
            }, ensure_ascii=False))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            res = predict(body.get("text", ""), body.get("generate", 0))
            self._send(json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self._send(json.dumps({"error": str(e)[:200]}, ensure_ascii=False), code=400)


def _open_browser(port, delay=1.5):
    time.sleep(delay)
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:%d" % port)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    load(args.checkpoint, args.device)
    print("loaded %s in %.1fs | %d domains | %s" %
          (os.path.basename(args.checkpoint), time.time() - t0, len(DOMAINS), DEV), flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("webui http://127.0.0.1:%d" % args.port, flush=True)
    if args.open:
        threading.Thread(target=_open_browser, args=(args.port,), daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
