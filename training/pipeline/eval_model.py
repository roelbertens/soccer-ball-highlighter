#!/usr/bin/env python3
"""Task-specific evaluation + leaderboard.

mAP is a poor proxy for "can grandpa follow the ball": what matters is
  - detection rate: frames where the predicted center is within R px of truth
  - center error in pixels (ring position quality)
  - false-ring rate on no-ball frames (very distracting!)
We report all of those at REALTIME settings (single scale, one model) plus
ultralytics mAP on the val split.

  highlight_score = det_rate - 0.5 * false_ring_rate      (higher = better)

Usage:
  python pipeline/eval_model.py --weights best_mex.pt --name baseline_960 --imgsz 960
  python pipeline/eval_model.py --weights runs/e2/weights/best.pt --match dui-par
Writes runs/leaderboard.json + LEADERBOARD.md.
"""
import argparse, os, sys, json, math, glob, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import ROOT, MATCHES, list_frames, read_label, frames_dir

LB_JSON = os.path.join(ROOT, 'runs', 'leaderboard.json')
LB_MD = os.path.join(ROOT, 'LEADERBOARD.md')


def eval_frames(model, pairs, imgsz, conf, device, radius_px):
    """pairs: list of (img_path, lab_or_None). lab None = confirmed no-ball frame."""
    import numpy as np
    hits, errs, n_pos, n_neg, false_rings, misses = 0, [], 0, 0, 0, 0
    for i, (img, lab) in enumerate(pairs, 1):
        if i % 50 == 0:
            print(f'   ...{i}/{len(pairs)}', flush=True)
        try:
            r = model.predict(img, imgsz=imgsz, conf=conf, device=device,
                              verbose=False)[0]
        except Exception:
            r = model.predict(img, imgsz=imgsz, conf=conf, device='cpu',
                              verbose=False)[0]
        H, W = r.orig_shape
        pred = None
        if r.boxes is not None and len(r.boxes):
            j = int(r.boxes.conf.argmax())
            x, y, w, h = [float(v) for v in r.boxes.xywhn[j].tolist()]
            pred = (x, y)
        if lab is None:
            n_neg += 1
            if pred:
                false_rings += 1
        else:
            n_pos += 1
            if pred:
                e = math.hypot((pred[0] - lab['cx']) * W, (pred[1] - lab['cy']) * H)
                errs.append(e)
                if e <= radius_px:
                    hits += 1
            else:
                misses += 1
    det_rate = hits / n_pos if n_pos else 0.0
    frr = false_rings / n_neg if n_neg else 0.0
    return {
        'n_pos': n_pos, 'n_neg': n_neg,
        'det_rate': round(det_rate, 4),
        'miss_rate': round(misses / n_pos, 4) if n_pos else 0.0,
        'false_ring_rate': round(frr, 4),
        'center_err_px_mean': round(sum(errs) / len(errs), 1) if errs else None,
        'center_err_px_median': round(sorted(errs)[len(errs) // 2], 1) if errs else None,
        'highlight_score': round(det_rate - 0.5 * frr, 4),
    }


def collect_pairs(match=None, split='val'):
    """From a match (all labeled frames + empty-label frames as negatives),
    or from the dataset split."""
    pairs = []
    if match:
        for f in list_frames(match):
            p = os.path.join(frames_dir(match), f)
            from bh_common import label_path
            lp = label_path(match, f)
            if not os.path.exists(lp):
                continue                      # never reviewed -> skip
            lab = read_label(match, f)
            pairs.append((p, lab))            # lab None = empty file = no ball
    else:
        img_dir = os.path.join(ROOT, 'data', 'dataset', 'images', split)
        lab_dir = os.path.join(ROOT, 'data', 'dataset', 'labels', split)
        for img in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
            lp = os.path.join(lab_dir, os.path.basename(img)[:-4] + '.txt')
            lab = None
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                q = open(lp).read().split()
                lab = {'cx': float(q[1]), 'cy': float(q[2]),
                       'w': float(q[3]), 'h': float(q[4])}
            pairs.append((img, lab))
    return pairs


def update_leaderboard(entry):
    rows = []
    if os.path.exists(LB_JSON):
        rows = json.load(open(LB_JSON))
    # dedup by (name, eval_on, imgsz) so one model can have a row per resolution
    def key(r):
        return (r['name'], r.get('eval_on', 'val'), r.get('imgsz'))
    rows = [r for r in rows if key(r) != key(entry)] + [entry]
    rows.sort(key=lambda r: -(r['metrics'].get('highlight_score') or 0))
    os.makedirs(os.path.dirname(LB_JSON), exist_ok=True)
    json.dump(rows, open(LB_JSON, 'w'), indent=1)
    cols = ['name', 'eval_on', 'imgsz', 'det_rate', 'center_err_px_median',
            'false_ring_rate', 'highlight_score', 'mAP50', 'date']
    lines = ['# Model leaderboard (higher highlight_score = better)', '',
             '| ' + ' | '.join(cols) + ' |',
             '|' + '---|' * len(cols)]
    for r in rows:
        mtr = r['metrics']
        lines.append('| ' + ' | '.join(str(x) for x in [
            r['name'], r.get('eval_on', 'val'), r.get('imgsz'),
            mtr.get('det_rate'), mtr.get('center_err_px_median'),
            mtr.get('false_ring_rate'), mtr.get('highlight_score'),
            mtr.get('mAP50'), r.get('date', '')[:16]]) + ' |')
    open(LB_MD, 'w').write('\n'.join(lines) + '\n')
    print(f'Leaderboard -> {LB_MD}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--imgsz', type=int, nargs='+', default=[800, 960, 1280],
                    help='eval size(s); one leaderboard row per size '
                         '(these match the extension deploy options)')
    ap.add_argument('--conf', type=float, default=0.10,
                    help='same default threshold as the extension')
    ap.add_argument('--radius-px', type=float, default=24.0)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--match', default=None, choices=MATCHES,
                    help='eval on all reviewed frames of one match instead of val split')
    ap.add_argument('--no-map', action='store_true', help='skip ultralytics val')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    pairs = collect_pairs(match=a.match)
    name = a.name or os.path.basename(os.path.dirname(
        os.path.dirname(a.weights)) or a.weights)
    sizes = a.imgsz if isinstance(a.imgsz, list) else [a.imgsz]
    map_res = 960 if 960 in sizes else sizes[0]   # mAP once, not per size
    out = {}
    for imgsz in sizes:
        print(f'Evaluating {a.weights} on {len(pairs)} frames '
              f'({a.match or "val split"}) @ imgsz={imgsz} conf={a.conf}')
        metrics = eval_frames(model, pairs, imgsz, a.conf, a.device, a.radius_px)
        if not a.no_map and not a.match and imgsz == map_res:
            try:
                v = model.val(data=os.path.join(ROOT, 'data', 'dataset', 'data.yaml'),
                              imgsz=imgsz, device=a.device, verbose=False)
                metrics['mAP50'] = round(float(v.box.map50), 4)
                metrics['mAP50_95'] = round(float(v.box.map), 4)
            except Exception as e:
                print('  (ultralytics val failed:', str(e)[:80], ')')
        print(json.dumps(metrics, indent=2))
        update_leaderboard({
            'name': name, 'weights': a.weights, 'imgsz': imgsz, 'conf': a.conf,
            'eval_on': a.match or 'val',
            'date': datetime.datetime.now().isoformat(), 'metrics': metrics})
        out[imgsz] = metrics
    return out


if __name__ == '__main__':
    main()
