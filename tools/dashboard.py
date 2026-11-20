"""零依赖训练监控面板（纯标准库 + 原生 canvas，不引 CDN）。

数据来自训练进程自己写的产物：
  heartbeat.json  最新指标（判活看心跳新鲜度，PID 只做辅助）
  train.log       每 100 步的训练曲线 + 每次验证的结果
  nvidia-smi      GPU 利用率 / 显存 / 温度（缓存 2 秒）

版式是工程控制台：深色命令栏、大数字看板、细线分区、单一强调色。
"""

import argparse
import ctypes
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

RUN = os.environ.get("RUN_DIR") or paths.run_dir()
HEARTBEAT = os.path.join(RUN, "heartbeat.json")
LAUNCH = os.path.join(RUN, "launch.json")
TRAIN_LOG = os.path.join(RUN, "train.log")

TRAIN_RE = re.compile(
    r"step\s+(\d+)\s+\|\s+([\d.]+)\s+min\s+\|\s+loss\s+([\d.]+)\s+"
    r"\(字\s+([\d.]+)\s+/\s+域\s+([\d.]+)\s+/\s+一致\s+([\d.]+)\s+/\s+对比\s+([\d.]+)\)\s+\|\s+"
    r"acc\s+([\d.]+)\s+\|\s+ent\s+([\d.]+)\s+\|\s+kl\s+([\d.]+)\s+\|\s+"
    r"lr\s+([\d.eE+-]+)\s+\|\s+([\d.]+)\s+字/秒\s+\|\s+显存\s+([\d.]+)G"
)
VALID_RE = re.compile(
    r"\[验证\]\s+step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+域\s+([\d.]+)\s+\|\s+"
    r"acc\s+([\d.]+)\s+\|\s+ent\s+([\d.]+)\s+\|\s+kl\s+([\d.]+)\s+\|\s+ppl\s+([\d.]+)\s+\|\s+"
    r"best\s+([\d.]+)@(\d+)\s+\|\s+连续未改善\s+(\d+)"
)
MILESTONE_RE = re.compile(r"(里程碑快照|早停|达到|结束)")

_GPU_CACHE = {"ts": 0.0, "data": {}}


def pid_alive(pid):
    if not pid:
        return False
    try:
        k = ctypes.windll.kernel32
        k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        k.OpenProcess.restype = ctypes.c_void_p
        k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k.WaitForSingleObject.restype = ctypes.c_ulong
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k.OpenProcess(0x00100000, False, int(pid))
        if not h:
            return False
        try:
            return k.WaitForSingleObject(h, 0) == 0x102
        finally:
            k.CloseHandle(h)
    except Exception:
        return False


def gpu_info():
    if time.time() - _GPU_CACHE["ts"] < 2.0:
        return _GPU_CACHE["data"]
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
        data = {"name": parts[0], "util": float(parts[1]), "mem_used": float(parts[2]),
                "mem_total": float(parts[3]), "temp": float(parts[4])}
    except Exception as e:
        data = {"error": str(e)[:80]}
    _GPU_CACHE["ts"] = time.time()
    _GPU_CACHE["data"] = data
    return data


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def translate(line):
    s = line.strip()
    if not s:
        return None
    m = TRAIN_RE.search(line)
    if m:
        return {"level": "info",
                "nice": "第 %s 步 · 总损失 %s（猜字 %s / 领域 %s / 一致 %s / 对比 %s）· 命中 %s · 已跑 %s 分钟" %
                        (m.group(1), m.group(3), m.group(4), m.group(5), m.group(6),
                         m.group(7), m.group(8), m.group(2)),
                "raw": s}
    v = VALID_RE.search(line)
    if v:
        return {"level": "ok",
                "nice": "验证 · 第 %s 步 · 命中 %s · KL %s · 熵 %s · 困惑度 %s · 最佳 %s@%s" %
                        (v.group(1), v.group(4), v.group(6), v.group(5), v.group(7),
                         v.group(8), v.group(9)),
                "raw": s}
    if re.search(r"Traceback|错误|失败|Error", s):
        level = "error"
    elif re.search(r"警告|注意|Warning", s):
        level = "warn"
    elif MILESTONE_RE.search(s):
        level = "ok"
    else:
        level = "info"
    return {"level": level, "nice": s, "raw": s}


def parse_log():
    series, valid, log = [], [], []
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return series, valid, log
    for line in lines[1:]:
        m = TRAIN_RE.search(line)
        if m:
            series.append({
                "step": int(m.group(1)), "min": float(m.group(2)),
                "loss": float(m.group(3)), "lm": float(m.group(4)),
                "dom": float(m.group(5)), "cons": float(m.group(6)),
                "ct": float(m.group(7)), "acc": float(m.group(8)),
                "ent": float(m.group(9)), "kl": float(m.group(10)),
                "lr": float(m.group(11)), "tps": float(m.group(12)),
                "vram": float(m.group(13)),
            })
            continue
        v = VALID_RE.search(line)
        if v:
            valid.append({
                "step": int(v.group(1)), "loss": float(v.group(2)),
                "dom": float(v.group(3)), "acc": float(v.group(4)),
                "ent": float(v.group(5)), "kl": float(v.group(6)),
                "ppl": float(v.group(7)), "best_kl": float(v.group(8)),
                "best_step": int(v.group(9)), "bad": int(v.group(10)),
            })
            log.append(translate(line))
            continue
        t = translate(line)
        if t:
            log.append(t)
    return series, valid, log[-40:]


def state():
    launch = read_json(LAUNCH) or {}
    hb = read_json(HEARTBEAT) or {}
    pid = launch.get("pid") or hb.get("pid")
    try:
        age = time.time() - os.path.getmtime(HEARTBEAT)
    except OSError:
        age = 1e9
    finished = bool(hb.get("finished")) or hb.get("alive") is False
    if finished:
        status = "finished"
    elif hb.get("alive") and age < 420:
        status = "running"
    elif hb:
        status = "stale"
    else:
        status = "stopped"
    series, valid, log = parse_log()
    return {
        "status": status, "pid": pid, "alive": pid_alive(pid),
        "heartbeat_age_seconds": round(age, 1), "heartbeat": hb,
        "launch": {k: launch.get(k) for k in ("started", "max_hours", "run_dir")},
        "gpu": gpu_info(), "series": series, "valid": valid, "log": log,
        "run_dir": RUN, "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练控制台 / TRAINING CONSOLE</title>
<style>
:root{
  --paper:#191919; --board:#222222; --ink:#EDEDED; --dim:#8A8A8A; --faint:#5A5A5A;
  --line:#333333; --accent:#FFFA00; --mag:#FF3B6B; --mint:#4DE1A0;
}
*{box-sizing:border-box;border-radius:0}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Saira","Bahnschrift","DIN Alternate","Segoe UI","Microsoft YaHei",sans-serif;
  font-size:13px;line-height:1.55;-webkit-font-smoothing:antialiased}
.num,.mono{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}
/* 命令栏 */
.bar{display:flex;align-items:center;gap:14px;height:46px;padding:0 20px;background:#0F0F0F;border-bottom:1px solid #000}
.bar .brand{font-size:12px;letter-spacing:.14em;font-stretch:125%;font-weight:700}
.bar .brand i{font-style:normal;color:var(--accent)}
.bar .sep{color:var(--faint)}
.bar .stat{font-size:11px;letter-spacing:.12em;font-stretch:125%;color:var(--dim)}
.bar .stat.run{color:var(--accent)}
.bar .spacer{flex:1}
.bar .clock{font-size:11px;color:var(--dim);letter-spacing:.08em}
/* 三色条：屏幕开始，只此一处 */
.stripe{display:flex;height:3px}
.stripe i{flex:1}
.stripe i:nth-child(1){background:var(--mag)}
.stripe i:nth-child(2){background:var(--accent)}
.stripe i:nth-child(3){background:var(--mint)}
main{padding:22px 20px 40px;max-width:1320px;margin:0 auto}
.eyebrow{font-size:10px;letter-spacing:.18em;font-stretch:125%;color:var(--dim);text-transform:uppercase}
h1{font-size:26px;font-weight:800;margin:2px 0 0;letter-spacing:.02em}
.head{display:flex;align-items:flex-end;gap:24px;flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--line)}
.fields{display:flex;gap:18px;flex-wrap:wrap;margin-left:auto;font-size:11px;color:var(--dim)}
.fields b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.fields span{display:flex;gap:6px;align-items:baseline}
.fields em{font-style:normal;color:var(--faint);font-size:10px;letter-spacing:.14em;font-stretch:125%}
/* 大数字看板 */
.board{display:grid;grid-template-columns:1.1fr 1.1fr 2fr;gap:0;border-bottom:1px solid var(--line)}
.big{padding:18px 20px 16px;border-right:1px solid var(--line);position:relative}
.big:last-child{border-right:0}
.big .n{font-size:52px;line-height:1;font-weight:800;font-stretch:125%;letter-spacing:-.01em}
.big .l{font-size:10px;letter-spacing:.18em;font-stretch:125%;color:var(--dim);margin-top:6px}
.big .s{font-size:11px;color:var(--faint);margin-top:2px}
.big.accent .n{color:var(--accent)}
.mini{display:grid;grid-template-columns:repeat(3,1fr);gap:0;padding:0}
.mini div{padding:12px 14px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.mini div:nth-child(3n){border-right:0}
.mini .k{font-size:9px;letter-spacing:.16em;font-stretch:125%;color:var(--faint);text-transform:uppercase}
.mini .v{font-size:19px;font-weight:700;margin-top:2px}
/* 图表区 */
.graphs{display:grid;grid-template-columns:1fr 1fr;gap:0;border-bottom:1px solid var(--line)}
@media(max-width:980px){.graphs{grid-template-columns:1fr}.board{grid-template-columns:1fr 1fr}.mini{grid-template-columns:repeat(2,1fr)}}
.g{padding:16px 18px 10px;border-right:1px solid var(--line)}
.g:last-child{border-right:0}
.g .t{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
.g .t b{font-size:14px;font-weight:700}
canvas{width:100%;height:210px;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px;font-size:10px;color:var(--dim);letter-spacing:.06em}
.legend s{display:inline-block;width:14px;height:2px;margin-right:5px;text-decoration:none;vertical-align:middle}
/* 表格 */
.sec{padding:16px 0 0}
.sec .t{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
.sec .t b{font-size:14px;font-weight:700}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font-size:9px;letter-spacing:.14em;font-stretch:125%;color:var(--faint);text-transform:uppercase;
  font-weight:500;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:7px 8px;text-align:right;border-bottom:1px solid #2a2a2a;font-size:12px;color:var(--ink)}
tr.best td:first-child{position:relative;padding-left:14px}
tr.best td:first-child:before{content:"";position:absolute;left:0;top:6px;bottom:6px;width:3px;background:var(--accent)}
.badge{display:inline-block;padding:1px 6px;background:var(--accent);color:#111;font-size:9px;
  letter-spacing:.12em;font-weight:700;font-stretch:125%}
/* 日志 */
.log{max-height:280px;overflow:auto;border-top:1px solid var(--line)}
.log div{display:flex;gap:12px;padding:6px 2px;border-bottom:1px solid #2a2a2a;font-size:12px;color:#C9C9C9}
.log div .idx{color:var(--faint);font-size:10px;letter-spacing:.1em;min-width:34px}
.log div.warn{background:repeating-linear-gradient(45deg,rgba(255,250,0,.06) 0 6px,transparent 6px 12px)}
.log div.warn .msg{color:var(--accent)}
.log div.error .msg{color:var(--accent);font-weight:700}
.log div.ok .msg{color:var(--ink)}
.log .raw{display:none;color:var(--faint);font-size:11px;margin-top:2px}
.log div:hover .raw{display:block}
.empty{padding:18px 4px;color:var(--faint);font-size:12px;background:repeating-linear-gradient(45deg,rgba(255,255,255,.03) 0 8px,transparent 8px 16px)}
.foot{margin-top:22px;padding-top:10px;border-top:1px solid var(--line);color:var(--faint);font-size:10px;letter-spacing:.1em}
@keyframes flash{0%{color:#111;background:var(--accent)}100%{color:inherit;background:transparent}}
.flash{animation:flash .38s cubic-bezier(.16,1,.3,1)}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
<div class="bar">
  <span class="brand">LLM<i>·</i>DOMAIN<i>·</i>LAB</span><span class="sep">/</span>
  <span class="stat" id="stat">CONNECTING</span>
  <span class="spacer"></span>
  <span class="clock mono" id="clock">--:--:--</span>
</div>
<div class="stripe"><i></i><i></i><i></i></div>
<main>
  <div class="head">
    <div>
      <div class="eyebrow">// TRAINING CONSOLE</div>
      <h1>训练控制台</h1>
    </div>
    <div class="fields" id="fields"></div>
  </div>

  <div class="board">
    <div class="big accent">
      <div class="n mono" id="vstep">0</div>
      <div class="l">STEP</div>
      <div class="s" id="vstepsub">上限 60,000</div>
    </div>
    <div class="big">
      <div class="n mono" id="vacc">—</div>
      <div class="l">VALID ACC</div>
      <div class="s">随机 3.3% / 30 类</div>
    </div>
    <div class="mini" id="mini"></div>
  </div>

  <div class="graphs">
    <div class="g">
      <div class="t"><div class="eyebrow">// LOSS</div><b>训练损失</b></div>
      <canvas id="c1"></canvas>
      <div class="legend">
        <span><s style="background:#FFFA00"></s>总损失</span>
        <span><s style="background:#EDEDED"></s>猜字</span>
        <span><s style="background:#8A8A8A"></s>领域</span>
        <span><s style="background:#5A5A5A"></s>一致 ×5</span>
        <span><s style="background:#3A3A3A"></s>对比 ×5</span>
      </div>
    </div>
    <div class="g">
      <div class="t"><div class="eyebrow">// VALIDATION</div><b>验证走势</b></div>
      <canvas id="c2"></canvas>
      <div class="legend">
        <span><s style="background:#FFFA00"></s>命中率</span>
        <span><s style="background:#EDEDED"></s>KL</span>
        <span><s style="background:#6A6A6A"></s>熵</span>
      </div>
    </div>
  </div>

  <div class="sec">
    <div class="t"><div class="eyebrow">// RECORDS</div><b>验证记录</b></div>
    <div id="vt"></div>
  </div>

  <div class="sec">
    <div class="t"><div class="eyebrow">// LOG</div><b>运行日志</b></div>
    <div class="log" id="log"></div>
  </div>

  <div class="foot" id="foot">只读训练产物 / 每 5 秒刷新 / 127.0.0.1:8901</div>
</main>
<script>
const $=id=>document.getElementById(id);
const pad=n=>String(n).padStart(2,'0');
function fmt(v,d){return v==null?'—':(typeof v==='number'?v.toFixed(d==null?3:d):v);}
function tickClock(){const d=new Date();$('clock').textContent=d.getFullYear()+'.'+pad(d.getMonth()+1)+'.'+pad(d.getDate())+'  '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());}
setInterval(tickClock,1000);tickClock();

function flash(el){el.classList.remove('flash');void el.offsetWidth;el.classList.add('flash');}

function drawChart(id,pts,series,opts){
  opts=opts||{};
  const c=$(id),dpr=window.devicePixelRatio||1;
  const w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
  const g=c.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  g.fillStyle='#191919';g.fillRect(0,0,w,h);
  g.strokeStyle='#2C2C2C';g.lineWidth=1;
  for(let i=0;i<=4;i++){const y=14+(h-34)*i/4;g.beginPath();g.moveTo(38,y);g.lineTo(w-6,y);g.stroke();}
  if(!pts||pts.length<2){
    g.fillStyle='#3A3A3A';g.font='11px Saira,sans-serif';g.fillText('WAITING FOR DATA',44,h/2);
    return;
  }
  let lo=Infinity,hi=-Infinity;
  for(const p of pts)for(const s of series){const v=p[s.key]*s.scale;if(v<lo)lo=v;if(v>hi)hi=v;}
  if(opts.zero)lo=Math.min(lo,0);
  if(hi-lo<1e-6)hi=lo+1;
  const pad2=(hi-lo)*0.08;lo-=pad2;hi+=pad2;
  const X=i=>38+(w-48)*i/(pts.length-1),Y=v=>14+(h-34)*(1-(v-lo)/(hi-lo));
  for(const s of series){
    g.strokeStyle=s.color;g.lineWidth=s.w||1.6;
    if(s.dash)g.setLineDash(s.dash);else g.setLineDash([]);
    g.beginPath();
    pts.forEach((p,i)=>{const xx=X(i),yy=Y(p[s.key]*s.scale);i?g.lineTo(xx,yy):g.moveTo(xx,yy);});
    g.stroke();
  }
  g.setLineDash([]);
  g.fillStyle='#5A5A5A';g.font='10px "Saira",Consolas,monospace';
  g.fillText(hi.toFixed(2),2,16);g.fillText(lo.toFixed(2),2,h-18);
  g.fillText('S'+pts[0].step,38,h-3);
  g.fillText('S'+pts[pts.length-1].step,w-58,h-3);
}

async function tick(){
  try{
    const s=await (await fetch('/api/state')).json();
    const h=s.heartbeat||{},g=s.gpu||{};
    const last=(s.series&&s.series.length)?s.series[s.series.length-1]:{};
    const lastV=(s.valid&&s.valid.length)?s.valid[s.valid.length-1]:{};
    const names={running:'RUNNING 训练中',finished:'FINISHED 已结束',stale:'STALE 异常中断',stopped:'IDLE 未启动'};
    const st=$('stat');st.textContent=names[s.status]||s.status;
    st.className='stat'+(s.status==='running'?' run':'');

    $('fields').innerHTML=
      '<span><em>LOSS</em><b>'+fmt(last.loss)+'</b></span>'+
      '<span><em>TOK/S</em><b>'+(last.tps?Math.round(last.tps).toLocaleString():'—')+'</b></span>'+
      '<span><em>LR</em><b>'+(last.lr!=null?last.lr.toExponential(2):'—')+'</b></span>'+
      '<span><em>VRAM</em><b>'+fmt(last.vram,2)+'G</b></span>'+
      '<span><em>GPU</em><b>'+(g.util??'—')+'%</b></span>'+
      '<span><em>TEMP</em><b>'+(g.temp??'—')+'C</b></span>';

    const stepNow=last.step||h.step||0;
    const vs=$('vstep');
    if(vs.textContent.replace(/,/g,'')!==String(stepNow)){vs.textContent=stepNow.toLocaleString();flash(vs);}
    $('vstepsub').textContent='上限 60,000 │ 已跑 '+((last.min||0).toFixed(0))+' 分';

    const acc=lastV.acc!=null?lastV.acc:h.valid_acc;
    const va=$('vacc');
    va.textContent=acc!=null?(acc*100).toFixed(1)+'%':'—';
    if(acc!=null&&acc>=0.5)va.style.color='#FFFA00';

    $('mini').innerHTML=
      '<div><div class="k">VALID KL</div><div class="v mono">'+fmt(lastV.kl!=null?lastV.kl:h.valid_kl)+'</div></div>'+
      '<div><div class="k">ENTROPY</div><div class="v mono">'+fmt(lastV.ent!=null?lastV.ent:h.valid_ent)+'</div></div>'+
      '<div><div class="k">PERPLEXITY</div><div class="v mono">'+fmt(lastV.ppl!=null?lastV.ppl:h.valid_ppl,1)+'</div></div>'+
      '<div><div class="k">BEST KL</div><div class="v mono">'+fmt(h.best_kl!=null?h.best_kl:lastV.best_kl)+'</div></div>'+
      '<div><div class="k">CONSISTENCY</div><div class="v mono">'+fmt(last.cons)+'</div></div>'+
      '<div><div class="k">CONTRAST</div><div class="v mono">'+fmt(last.ct)+'</div></div>';

    drawChart('c1',(s.series||[]).slice(-400),[
      {key:'loss',color:'#FFFA00',scale:1,w:2},
      {key:'lm',color:'#EDEDED',scale:1},
      {key:'dom',color:'#8A8A8A',scale:1},
      {key:'cons',color:'#5A5A5A',scale:5},
      {key:'ct',color:'#3A3A3A',scale:5,dash:[3,3]}],{zero:true});
    drawChart('c2',s.valid||[],[
      {key:'acc',color:'#FFFA00',scale:1,w:2},
      {key:'kl',color:'#EDEDED',scale:1},
      {key:'ent',color:'#6A6A6A',scale:1}],{zero:true});

    const vt=(s.valid||[]).slice().reverse().slice(0,12);
    $('vt').innerHTML=vt.length?
      '<table><tr><th>STEP</th><th>验证损失</th><th>领域</th><th>命中</th><th>熵</th><th>KL</th><th>困惑度</th><th>最佳</th></tr>'+
      vt.map(v=>'<tr'+(v.best_step===v.step?' class="best"':'')+'><td class="mono">'+v.step.toLocaleString()+'</td><td class="mono">'+v.loss.toFixed(3)+'</td><td class="mono">'+v.dom.toFixed(3)+'</td><td class="mono">'+(v.acc*100).toFixed(1)+'%</td><td class="mono">'+v.ent.toFixed(3)+'</td><td class="mono">'+v.kl.toFixed(3)+'</td><td class="mono">'+v.ppl.toFixed(0)+'</td><td class="mono">'+(v.best_step===v.step?'★ ':'')+v.best_step.toLocaleString()+'</td></tr>').join('')+'</table>'
      :'<div class="empty">还没有验证记录。训练每 1,500 步评一次验证集，到时这里会出现第一行。</div>';

    const lg=(s.log||[]).slice().reverse();
    $('log').innerHTML=lg.length?lg.map((e,i)=>
      '<div class="'+e.level+'"><span class="idx">'+pad(lg.length-i)+'</span><span><span class="msg">'+e.nice.replace(/</g,'&lt;')+'</span><span class="raw">'+e.raw.replace(/</g,'&lt;')+'</span></span></div>').join('')
      :'<div class="empty">日志会在训练写出第一行后出现。</div>';

    $('foot').textContent='RUN DIR  '+s.run_dir+'   /   心跳 '+s.heartbeat_age_seconds+' 秒前   /   只读产物，每 5 秒刷新';
  }catch(e){
    $('stat').textContent='DISCONNECTED';
    $('foot').textContent='读取失败：'+e;
  }
}
window.addEventListener('resize',()=>tick());
setInterval(tick,5000);tick();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(json.dumps(state(), ensure_ascii=False))
        elif self.path == "/" or self.path.startswith("/index"):
            self._send(HTML, "text/html; charset=utf-8")
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("dashboard http://127.0.0.1:%d  run_dir=%s" % (args.port, RUN), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
