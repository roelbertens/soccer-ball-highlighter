#!/usr/bin/env python3
# Run the (snapshot) model over the mex-ecu frames and collect wrong/unsure
# frames vs the (corrected) labels. Output: data/errors_mex.json for the
# review tool. (Superseded by pipeline/refine_labels.py, kept for reference.)
from ultralytics import YOLO
import os, json, glob, math

FRAMES = 'data/frames/mex-ecu'
LABELS = 'data/labels/mex-ecu/labels'
CONF = 0.08          # low: also show weak (possibly wrong) boxes
LOC_THR = 0.03       # normalized distance above which a box is "off target"

model = YOLO('best_mex.pt')
imgs = sorted(glob.glob(FRAMES + '/*.jpg'))

def read_lab(name):
    p = os.path.join(LABELS, name + '.txt')
    if os.path.exists(p) and os.path.getsize(p) > 0:
        q = open(p).read().split()
        return {'cx': float(q[1]), 'cy': float(q[2]), 'w': float(q[3]), 'h': float(q[4])}
    return None

items = []
IMGSZ = 960          # a bit lower: lighter on memory + avoids the MPS buffer bug
for done, img in enumerate(imgs, 1):
    if done % 50 == 0:
        print(f'  ...{done}/{len(imgs)}  ({len(items)} errors so far)', flush=True)
        json.dump(items, open('data/errors_mex.json', 'w'))   # checkpoint
    name = os.path.basename(img)
    try:
        r = model.predict(img, imgsz=IMGSZ, conf=CONF, device='cpu', verbose=False)[0]
        pred = None
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x, y, w, h = [float(v) for v in r.boxes.xywhn[i].tolist()]
            pred = {'cx': x, 'cy': y, 'w': w, 'h': h, 'conf': float(r.boxes.conf[i])}
    except Exception as e:
        print('  frame skipped', name, str(e)[:60], flush=True)
        continue
    lab = read_lab(name)
    reason = None
    if pred and not lab:
        reason = 'fp'                                   # box drawn where label says 'no ball'
    elif pred and lab:
        d = math.hypot(pred['cx'] - lab['cx'], pred['cy'] - lab['cy'])
        if d > LOC_THR:
            reason = 'loc'                              # box is off target
    elif lab and not pred:
        reason = 'miss'                                 # misses a labeled ball
    if reason:
        items.append({'file': name, 'reason': reason, 'pred': pred, 'lab': lab})

# sort: false positives first (wrong boxes), then localization, then misses
order = {'fp': 0, 'loc': 1, 'miss': 2}
items.sort(key=lambda it: (order[it['reason']], -(it['pred']['conf'] if it['pred'] else 0)))
json.dump(items, open('data/errors_mex.json', 'w'))
from collections import Counter
c = Counter(it['reason'] for it in items)
print(f"{len(items)} error frames -> fp(wrong box)={c['fp']}  loc(off target)={c['loc']}  miss(missed)={c['miss']}")
