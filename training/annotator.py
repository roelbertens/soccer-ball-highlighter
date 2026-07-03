#!/usr/bin/env python3
# Local ball-annotation tool with 2 modes:
#  - "check labels": step through frames, correct/delete the auto labels.
#  - "review queue": step only through frames where the model is wrong/unsure
#    (pink = model prediction, cyan = your label). X = no ball -> negative.
# Shortcuts: space/-> = correct · click = ball here · x = no ball · <- = back.
# Writes YOLO labels back + keeps a negatives list (used as training background).
import http.server, socketserver, json, os, urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(ROOT, 'data/frames')
LABELS = os.path.join(ROOT, 'data/labels')
MATCHES = ['mex-ecu', 'ivo-nor', 'fra-zwe', 'dui-par']
PORT = int(os.environ.get('BH_PORT', 8000))

def lp(match, imgfile):
    return os.path.join(LABELS, match, 'labels', imgfile + '.txt')

def negfile(match):
    return os.path.join(LABELS, match, 'negatives.txt')

def read_label(match, f):
    p = lp(match, f)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        parts = open(p).read().split()
        if len(parts) >= 5:
            return {'cx': float(parts[1]), 'cy': float(parts[2]), 'w': float(parts[3]), 'h': float(parts[4])}
    return None

def build_items(match, only_labeled):
    items = []
    ms = MATCHES if match == 'all' else [match]
    for m in ms:
        d = os.path.join(FRAMES, m)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.jpg'):
                continue
            lab = read_label(m, f)
            if only_labeled and lab is None:
                continue
            items.append({'match': m, 'file': f, 'label': lab})
    return items

def read_negs(match):
    p = negfile(match)
    if os.path.exists(p):
        return set(l.strip() for l in open(p) if l.strip())
    return set()

def write_neg(match, f, add):
    negs = read_negs(match)
    if add:
        negs.add(f)
    else:
        negs.discard(f)
    os.makedirs(os.path.dirname(negfile(match)), exist_ok=True)
    open(negfile(match), 'w').write('\n'.join(sorted(negs)) + ('\n' if negs else ''))

HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>Ball annotation</title>
<style>
 body{margin:0;background:#111;color:#eee;font:14px system-ui;overflow:hidden}
 #start{padding:30px;max-width:680px;margin:auto}
 button{background:#00e5ff;color:#003;border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer;margin:4px}
 button.err{background:#ff2bd6;color:#fff}
 #wrap{position:fixed;inset:0;display:none}
 #stage{position:absolute;inset:0 0 44px 0;display:flex;align-items:center;justify-content:center}
 #img{max-width:100%;max-height:100%;cursor:crosshair;user-select:none}
 #bar{position:absolute;left:0;right:0;bottom:0;height:44px;background:#000a;display:flex;align-items:center;gap:16px;padding:0 14px;font-size:13px}
 .bx{position:absolute;pointer-events:none;box-shadow:0 0 0 1px #000}
 #box{border:2px solid #00e5ff}
 #pbox{border:2px solid #ff2bd6}
 #loupe{position:fixed;width:180px;height:180px;border:2px solid #00e5ff;border-radius:50%;pointer-events:none;display:none;background-repeat:no-repeat;box-shadow:0 4px 16px #000b;z-index:9}
 #loupe::after{content:"";position:absolute;left:50%;top:50%;width:10px;height:10px;margin:-5px;border:1px solid #ff2bd6;border-radius:50%}
 .k{color:#00e5ff;font-weight:700}
 .badge{padding:2px 8px;border-radius:4px;font-weight:700}
 .fp{background:#ff2bd6;color:#fff}.loc{background:#ffb020;color:#000}.miss{background:#666;color:#fff}
 #prog{height:4px;background:#00e5ff;position:absolute;left:0;top:0}
</style></head><body>
<div id=start>
 <h2>⚽ Ball annotation</h2>
 <p><b>Shortcuts:</b> <span class=k>space</span>/<span class=k>→</span> = correct · <span class=k>click</span> = ball here · <span class=k>x</span> = no ball · <span class=k>←</span> = back</p>
 <h3>1) Check labels (correct the auto labels)</h3>
 <div id=btns></div>
 <label style="display:block;margin-top:8px"><input type=checkbox id=onlylab checked> only frames WITH a label</label>
 <h3 style="margin-top:22px">2) Review queues (where the model disagrees)</h3>
 <p style="color:#aaa">pink box = what the model predicts · cyan = your label. Too hard? Press <span class=k>x</span> → trained as "no ball".</p>
 <div id=errbtns></div>
</div>
<div id=wrap>
 <div id=prog></div>
 <div id=stage><img id=img><div id=box class=bx></div><div id=pbox class=bx></div></div>
 <div id=bar>
   <span id=reason></span>
   <span id=info></span>
   <span id=count></span>
   <span style="margin-left:auto"><span class=k>space</span>=correct <span class=k>click</span>=ball <span class=k>x</span>=no&nbsp;ball <span class=k>←</span>=back</span>
 </div>
</div>
<div id=loupe></div>
<script>
let items=[], idx=0, scope='', mode='items';
const img=document.getElementById('img'), boxEl=document.getElementById('box'),
      pboxEl=document.getElementById('pbox'), loupe=document.getElementById('loupe');
const BOXPX=24;

const btns=document.getElementById('btns');
['all','mex-ecu','ivo-nor','fra-zwe','dui-par'].forEach(m=>{
 const b=document.createElement('button');b.textContent=m;b.onclick=()=>start(m);btns.appendChild(b);});
const eb=document.getElementById('errbtns');
['mex-ecu','ivo-nor','fra-zwe','dui-par'].forEach(m=>{
 const b=document.createElement('button');b.className='err';b.textContent='🔴 '+m;
 b.onclick=()=>start('errors:'+m);eb.appendChild(b);});

async function start(what){
 if(what==='errors'||what.startsWith('errors:')){
   const m=what.includes(':')?what.split(':')[1]:'mex-ecu';
   mode='errors'; scope='errors:'+m;
   const r=await fetch('/api/errors?match='+encodeURIComponent(m)); items=await r.json();
   if(!items.length){alert('No review items found (or analysis not done yet).');return;}
 }else{
   mode='items'; const only=document.getElementById('onlylab').checked?1:0;
   scope=what+'|'+only;
   const r=await fetch('/api/items?match='+encodeURIComponent(what)+'&only='+only);
   items=await r.json();
 }
 idx=+(localStorage.getItem('idx_'+scope)||0); if(idx>=items.length)idx=0;
 document.getElementById('start').style.display='none';
 document.getElementById('wrap').style.display='block';
 show();
}
function cur(){const it=items[idx]; it.match=it.match||'mex-ecu'; return it;}
function show(){
 if(idx<0)idx=0;
 if(idx>=items.length){alert('Done! '+items.length+' frames reviewed.');return;}
 localStorage.setItem('idx_'+scope,idx);
 const it=cur();
 img.onload=drawBoxes;
 img.src='/img/'+it.match+'/'+it.file+'?_='+idx;
 let rb=document.getElementById('reason');
 if(mode==='errors'){const R={fp:'wrong box',loc:'off target',miss:'missed'};
   rb.innerHTML='<span class="badge '+it.reason+'">'+R[it.reason]+(it.pred?(' '+Math.round(it.pred.conf*100)+'%'):'')+'</span>';}
 else rb.textContent='';
 document.getElementById('info').textContent=it.match+' · '+it.file;
 document.getElementById('count').textContent=(idx+1)+' / '+items.length;
 document.getElementById('prog').style.width=(100*(idx+1)/items.length)+'%';
 for(let i=1;i<=2;i++){if(idx+i<items.length){const im=new Image();const n=items[idx+i];im.src='/img/'+(n.match||'mex-ecu')+'/'+n.file+'?_='+(idx+i);}}
}
function place(el,bx){
 const r=img.getBoundingClientRect();
 if(!bx){el.style.display='none';return;}
 const w=Math.max(10,(bx.w||0.012)*r.width), h=Math.max(10,(bx.h||0.02)*r.height);
 el.style.display='block';
 el.style.left=(r.left+bx.cx*r.width-w/2)+'px';
 el.style.top=(r.top+bx.cy*r.height-h/2)+'px';
 el.style.width=w+'px'; el.style.height=h+'px';
}
function drawBoxes(){
 const it=cur();
 place(boxEl, it.label||it.lab||null);         // cyan = current label
 place(pboxEl, mode==='errors'?(it.pred||null):null);  // pink = model prediction
}
async function save(label, neg){
 const it=cur(); it.label=label; if('lab' in it) it.lab=label;
 await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({match:it.match,file:it.file,label:label,neg:!!neg})});
}
function next(){idx++;show();}
function prev(){idx--;show();}

img.addEventListener('click',async e=>{
 const r=img.getBoundingClientRect();
 const cx=(e.clientX-r.left)/r.width, cy=(e.clientY-r.top)/r.height;
 const w=BOXPX/img.naturalWidth, h=BOXPX/img.naturalHeight;
 await save({cx,cy,w,h}, false); next();     // click = positive ball, clears the negative flag
});
img.addEventListener('mousemove',e=>{
 const r=img.getBoundingClientRect(), zoom=4, L=180;
 loupe.style.display='block';
 loupe.style.left=(e.clientX-L/2)+'px'; loupe.style.top=(e.clientY-L-16)+'px';
 loupe.style.backgroundImage="url('"+img.src+"')";
 loupe.style.backgroundSize=(r.width*zoom)+'px '+(r.height*zoom)+'px';
 loupe.style.backgroundPosition=(-((e.clientX-r.left)*zoom-L/2))+'px '+(-((e.clientY-r.top)*zoom-L/2))+'px';
});
img.addEventListener('mouseleave',()=>loupe.style.display='none');
window.addEventListener('resize',drawBoxes);
document.addEventListener('keydown',async e=>{
 if(document.getElementById('wrap').style.display==='none')return;
 if(e.key===' '||e.key==='ArrowRight'){e.preventDefault();next();}
 else if(e.key==='ArrowLeft'){e.preventDefault();prev();}
 else if(e.key==='x'||e.key==='Delete'||e.key==='Backspace'){e.preventDefault();
   await save(null, true); next();}   // x = no ball => always train as negative
});
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        self.send_response(code); self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ('/', '/index.html'):
            return self._send(200, HTML.encode('utf-8'), 'text/html; charset=utf-8')
        if u.path == '/api/items':
            items = build_items(q.get('match', ['all'])[0], q.get('only', ['1'])[0] == '1')
            return self._send(200, json.dumps(items).encode())
        if u.path == '/api/errors':
            m = q.get('match', ['mex-ecu'])[0]
            p = os.path.join(ROOT, 'data/review', m + '.json')
            if not os.path.exists(p) and m == 'mex-ecu':
                p = os.path.join(ROOT, 'data/errors_mex.json')   # legacy location
            data = json.load(open(p)) if os.path.exists(p) else []
            for it in data:
                it['match'] = m
            return self._send(200, json.dumps(data).encode())
        if u.path.startswith('/img/'):
            rel = urllib.parse.unquote(u.path[len('/img/'):])
            if '/' not in rel or '..' in rel:
                return self._send(404, b'no')
            m, f = rel.split('/', 1)
            p = os.path.join(FRAMES, m, f)
            if os.path.exists(p):
                return self._send(200, open(p, 'rb').read(), 'image/jpeg')
            return self._send(404, b'no')
        return self._send(404, b'no')

    def do_POST(self):
        if self.path == '/api/save':
            n = int(self.headers.get('Content-Length', 0))
            d = json.loads(self.rfile.read(n))
            p = lp(d['match'], d['file'])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            lab = d.get('label')
            with open(p, 'w') as fh:
                if lab:
                    fh.write(f"0 {lab['cx']:.6f} {lab['cy']:.6f} {lab['w']:.6f} {lab['h']:.6f}\n")
            # update the negatives list: x => negative; click => removed from list
            write_neg(d['match'], d['file'], bool(d.get('neg')) and lab is None)
            return self._send(200, b'{"ok":true}')
        return self._send(404, b'no')

    def log_message(self, *a):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', PORT), H) as httpd:
    print(f"Annotation tool running at http://localhost:{PORT}")
    httpd.serve_forever()
