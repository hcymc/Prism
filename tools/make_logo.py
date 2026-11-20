"""生成 Prism 的 1:1 logo：SVG 源文件 + 多尺寸 PNG + favicon.ico。

设计约束：平面、克制。纯几何，一个墨色 + 一个强调色，没有渐变、阴影、辉光；
刻意避开「棱镜 + 白光 + 彩虹扇」（Pink Floyd《The Dark Side of the Moon》）。

候选方向（--variant）：
  cut     实心三角被水平切开，切下的那一条整体平移出去、染成强调色
          —— 31 个领域里被「抽出来」的那一个
  peak    三角的顶尖与主体分开、染成强调色 —— 分布的峰值
  slit    实心三角上开一道细缝，缝里透出强调色、向右穿出

主题（--theme）：dark（深底黄标，endfield 配色）/ light（浅底墨标）

    python tools/make_logo.py                          # 默认 → assets/logo/
    python tools/make_logo.py --compare                # 所有方向 × 两种主题的对比图
    python tools/make_logo.py --variant peak --theme light

PNG 用本机 Edge / Chrome 的 headless 截图渲染 1024px 母版，再用 Pillow 缩到各尺寸。
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "assets", "logo")
SIZES = [16, 32, 48, 64, 128, 180, 256, 512, 1024]

THEMES = {
    #         底色        主体        强调
    "dark":  ("#161616", "#ededed", "#fffa00"),
    "light": ("#f4f4f2", "#161616", "#ff4d00"),
}

# 正三角形，视觉居中（几何重心略低于画布中心，看起来才是正的）
S = 300                         # 边长
H = S * math.sqrt(3) / 2
CX, TOP = 256, 256 - H * 0.56
BOT = TOP + H


def half_w(y):
    return S / 2 * (y - TOP) / H


def tri_band(y0, y1, dx=0.0):
    """三角形在 [y0, y1] 之间那一段（顶上是三角、其余是梯形），可整体右移 dx。"""
    a, b = half_w(y0), half_w(y1)
    return "%.2f,%.2f %.2f,%.2f %.2f,%.2f %.2f,%.2f" % (
        CX - a + dx, y0, CX + a + dx, y0, CX + b + dx, y1, CX - b + dx, y1)


def frame(bg, body):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" '
        'role="img" aria-label="Prism logo">\n  <title>Prism</title>\n'
        '  <rect width="512" height="512" rx="96" fill="%s"/>\n%s</svg>\n' % (bg, body)
    )


def v_cut(theme):
    bg, ink, acc = THEMES[theme]
    gap = 10
    y0 = TOP + H * 0.58          # 切下来那一条的上沿
    y1 = y0 + H * 0.11           # 下沿
    dx = 64
    o = -dx * 0.3                # 整体左移一点，抵消切片右移造成的重心偏右
    body = (
        '  <polygon points="%s" fill="%s"/>\n' % (tri_band(TOP, y0 - gap, o), ink)
        + '  <polygon points="%s" fill="%s"/>\n' % (tri_band(y1 + gap, BOT, o), ink)
        + '  <polygon points="%s" fill="%s"/>\n' % (tri_band(y0, y1, o + dx), acc)
    )
    return frame(bg, body)


def v_peak(theme):
    bg, ink, acc = THEMES[theme]
    gap = 16
    yk = TOP + H * 0.30
    body = (
        '  <polygon points="%s" fill="%s"/>\n' % (tri_band(TOP, yk), acc)
        + '  <polygon points="%s" fill="%s"/>\n' % (tri_band(yk + gap, BOT), ink)
    )
    return frame(bg, body)


def v_slit(theme):
    bg, ink, acc = THEMES[theme]
    y0 = TOP + H * 0.60
    y1 = y0 + 14
    body = (
        '  <polygon points="%s" fill="%s"/>\n' % (tri_band(TOP, y0), ink)
        + '  <polygon points="%s" fill="%s"/>\n' % (tri_band(y1, BOT), ink)
        + '  <rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>\n'
        % (CX - half_w(y1), y0, 2 * half_w(y1) + 56, y1 - y0, acc)
    )
    return frame(bg, body)


VARIANTS = {"cut": v_cut, "peak": v_peak, "slit": v_slit}
DEFAULT = ("cut", "dark")


def find_browser():
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    return None


def render_html(inner, out_png, w, h, bg="transparent"):
    browser = find_browser()
    if not browser:
        raise RuntimeError("找不到 Edge / Chrome，无法渲染 PNG；可用 --no-png 只出 SVG")
    tmp = tempfile.mkdtemp(prefix="prism_logo_")
    try:
        html = os.path.join(tmp, "logo.html")
        with open(html, "w", encoding="utf-8") as f:
            f.write("<!doctype html><html><head><style>html,body{margin:0;background:%s}"
                    "</style></head><body>%s</body></html>" % (bg, inner))
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", "--default-background-color=00000000",
             "--user-data-dir=" + os.path.join(tmp, "profile"),
             "--window-size=%d,%d" % (w, h),
             "--screenshot=" + out_png, "file:///" + html.replace("\\", "/")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sized(svg, px):
    return svg.replace('width="512" height="512"', 'width="%d" height="%d" style="display:block"' % (px, px), 1)


def compare(out_dir):
    """每个方向一行：深色 / 浅色 各 200px + 48 + 24，放在中灰底上看两种主题。"""
    rows = []
    for name, fn in VARIANTS.items():
        cells = []
        for theme in ("dark", "light"):
            svg = fn(theme)
            cells.append("".join('<div style="display:flex;align-items:center;justify-content:center;width:%dpx">%s</div>'
                                 % (200 if s == 200 else 56, sized(svg, s)) for s in (200, 48, 24)))
        rows.append('<div style="display:flex;align-items:center;gap:20px;padding:14px 20px;'
                    'font:600 18px sans-serif;color:#161616"><div style="width:52px">%s</div>%s'
                    '<div style="width:16px"></div>%s</div>' % (name, cells[0], cells[1]))
    p = os.path.join(out_dir, "compare.png")
    render_html("".join(rows), p, 900, len(VARIANTS) * 228 + 8, bg="#c9c9c6")
    print("cmp ", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT[0])
    ap.add_argument("--theme", choices=sorted(THEMES), default=DEFAULT[1])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.compare:
        compare(args.out)
        return

    svg = VARIANTS[args.variant](args.theme)
    svg_path = os.path.join(args.out, "prism-logo.svg")
    with open(svg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    print("svg ", svg_path, "(%s / %s)" % (args.variant, args.theme))
    if args.no_png:
        return

    from PIL import Image

    master = os.path.join(args.out, "prism-logo-1024.png")
    render_html(sized(svg, 1024), master, 1024, 1024)
    img = Image.open(master).convert("RGBA")
    if img.size != (1024, 1024):
        sys.exit("母版尺寸不对：%s（检查系统缩放）" % (img.size,))
    for s in SIZES:
        p = os.path.join(args.out, "prism-logo-%d.png" % s)
        if s != 1024:
            img.resize((s, s), Image.LANCZOS).save(p, optimize=True)
        print("png ", p)
    ico = os.path.join(args.out, "favicon.ico")
    img.save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("ico ", ico)


if __name__ == "__main__":
    main()
