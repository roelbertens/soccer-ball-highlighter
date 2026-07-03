#!/usr/bin/env python3
"""Shared helpers for the ball-highlighter data pipeline.

Conventions (same as annotator.py / build_dataset.py):
  frames:  data/frames/<match>/<match>_NNNN.jpg
  labels:  data/labels/<match>/labels/<match>_NNNN.jpg.txt   (YOLO, 1 line, class 0)
  negs:    data/labels/<match>/negatives.txt                  (frame filenames, no ball)
  review:  data/review/<match>.json                           (queue for annotator.py)
"""
import os, json, glob, math, shutil, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ball-highlighter-build
FRAMES = os.path.join(ROOT, 'data', 'frames')
LABELS = os.path.join(ROOT, 'data', 'labels')
REVIEW = os.path.join(ROOT, 'data', 'review')
VIDEOS = os.path.join(ROOT, 'videos')
MATCHES = ['mex-ecu', 'ivo-nor', 'fra-zwe', 'dui-par']

# Manual clicks in annotator.py always produce a fixed 24px box (at 1920x1078):
FIXED_W, FIXED_H = 0.012500, 0.022263


def frames_dir(match):
    return os.path.join(FRAMES, match)


def list_frames(match):
    d = frames_dir(match)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith('.jpg'))


def label_path(match, imgfile):
    return os.path.join(LABELS, match, 'labels', imgfile + '.txt')


def read_label(match, imgfile):
    p = label_path(match, imgfile)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        parts = open(p).read().split()
        if len(parts) >= 5:
            return {'cx': float(parts[1]), 'cy': float(parts[2]),
                    'w': float(parts[3]), 'h': float(parts[4])}
    return None


def has_label_file(match, imgfile):
    return os.path.exists(label_path(match, imgfile))


def write_label(match, imgfile, lab):
    """lab=None writes an empty file (= reviewed, no ball)."""
    p = label_path(match, imgfile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as fh:
        if lab:
            fh.write(f"0 {lab['cx']:.6f} {lab['cy']:.6f} {lab['w']:.6f} {lab['h']:.6f}\n")


def read_negatives(match):
    p = os.path.join(LABELS, match, 'negatives.txt')
    if os.path.exists(p):
        return set(l.strip() for l in open(p) if l.strip())
    return set()


def is_fixed_size(lab, tol=1e-4):
    return (lab and abs(lab['w'] - FIXED_W) < tol and abs(lab['h'] - FIXED_H) < tol)


def dist(a, b):
    return math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy'])


def backup_labels(match):
    src = os.path.join(LABELS, match, 'labels')
    if not os.path.isdir(src):
        return None
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(LABELS, match, f'labels_backup_{stamp}')
    shutil.copytree(src, dst)
    return dst


def save_review(match, items, merge=True):
    """Write review queue for annotator.py (same schema as errors_mex.json).
    Merges with an existing queue (dedup by file, new item wins) so that
    refine_labels + autolabel don't clobber each other's queues."""
    os.makedirs(REVIEW, exist_ok=True)
    p = os.path.join(REVIEW, match + '.json')
    if merge and os.path.exists(p):
        try:
            old = {it['file']: it for it in json.load(open(p))}
        except Exception:
            old = {}
        for it in items:
            old[it['file']] = it
        items = list(old.values())
        order = {'fp': 0, 'loc': 1, 'miss': 2}
        items.sort(key=lambda it: order.get(it.get('reason'), 3))
    json.dump(items, open(p, 'w'))
    return p


# ---------------------------------------------------------------- detection

class BallDetector:
    """Expensive offline ensemble: fine-tuned model + big COCO model, multi-scale,
    with 2x2 tiled fallback for tiny balls. NOT for realtime - for labeling only."""

    def __init__(self, weights=None, imgszs=(1280, 1920), device='mps',
                 conf=0.05, use_tiles=True):
        from ultralytics import YOLO  # lazy import
        self.models = []
        weights = weights or self._default_weights()
        for w in weights:
            m = YOLO(w)
            cls = self._ball_class(m)
            if cls is None:
                print(f'  !! {w}: no ball class found, skipping')
                continue
            self.models.append((os.path.basename(w), m, cls))
        if not self.models:
            raise SystemExit('No usable models found.')
        self.imgszs = tuple(imgszs)
        self.device = device
        self.conf = conf
        self.use_tiles = use_tiles

    @staticmethod
    def _default_weights():
        ws = []
        for cand in ['best_mex.pt', 'soccer_ball.pt']:
            p = os.path.join(ROOT, cand)
            if os.path.exists(p):
                ws.append(p)
                break
        # big generic model as second opinion (auto-downloads if missing)
        ws.append('yolov8x.pt')
        return ws

    @staticmethod
    def _ball_class(model):
        for i, n in model.names.items():
            if 'ball' in str(n).lower():   # 'ball' or 'sports ball'
                return int(i)
        return None

    def _predict_raw(self, source, model, cls, imgsz):
        try:
            r = model.predict(source, imgsz=imgsz, conf=self.conf,
                              device=self.device, verbose=False)[0]
        except Exception:
            r = model.predict(source, imgsz=imgsz, conf=self.conf,
                              device='cpu', verbose=False)[0]
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            if int(b.cls) != cls:
                continue
            cx, cy, w, h = [float(v) for v in b.xywhn[0].tolist()]
            out.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'conf': float(b.conf)})
        return out

    def _tiles(self, img_path, model, cls, imgsz):
        """2x2 overlapping tiles -> normalized full-frame coords."""
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return []
        H, W = img.shape[:2]
        tw, th = int(W * 0.6), int(H * 0.6)   # 60% tiles -> 20% overlap
        out = []
        for ox in (0, W - tw):
            for oy in (0, H - th):
                crop = img[oy:oy + th, ox:ox + tw]
                for d in self._predict_raw(crop, model, cls, imgsz):
                    out.append({'cx': (ox + d['cx'] * tw) / W,
                                'cy': (oy + d['cy'] * th) / H,
                                'w': d['w'] * tw / W, 'h': d['h'] * th / H,
                                'conf': d['conf'] * 0.95})  # slight penalty
        return out

    def detect(self, img_path):
        """Returns fused detections sorted by conf desc."""
        dets = []
        for name, model, cls in self.models:
            for sz in self.imgszs:
                for d in self._predict_raw(img_path, model, cls, sz):
                    d['src'] = f'{name}@{sz}'
                    dets.append(d)
        if not dets and self.use_tiles:
            name, model, cls = self.models[0]
            for d in self._tiles(img_path, model, cls, self.imgszs[0]):
                d['src'] = f'{name}@tile'
                dets.append(d)
        return self._fuse(dets)

    @staticmethod
    def _fuse(dets, radius=0.02):
        """Cluster detections by center distance; conf-weighted merge per cluster."""
        clusters = []
        for d in sorted(dets, key=lambda x: -x['conf']):
            for c in clusters:
                if dist(c[0], d) < radius:
                    c.append(d)
                    break
            else:
                clusters.append([d])
        fused = []
        for c in clusters:
            wsum = sum(d['conf'] for d in c)
            f = {k: sum(d[k] * d['conf'] for d in c) / wsum
                 for k in ('cx', 'cy', 'w', 'h')}
            # agreement between independent sources boosts confidence
            nsrc = len(set(d['src'].split('@')[0] for d in c))
            f['conf'] = min(0.99, max(d['conf'] for d in c) * (1.0 + 0.15 * (nsrc - 1)))
            f['nsrc'] = nsrc
            f['src'] = '+'.join(sorted(set(d['src'] for d in c)))
            fused.append(f)
        return sorted(fused, key=lambda x: -x['conf'])


# ---------------------------------------------------------------- temporal

def temporal_flag(prev, cur, nxt, jump=0.08):
    """True if cur jumps away from BOTH temporally-adjacent positions while
    those two agree with each other (classic outlier pattern)."""
    if not (prev and cur and nxt):
        return False
    return (dist(prev, cur) > jump and dist(cur, nxt) > jump
            and dist(prev, nxt) < jump)


def interpolate(prev, nxt, frac=0.5, max_move=0.10):
    """Midpoint interpolation, only when neighbors are close enough that the
    ball plausibly moved linearly (frames are ~0.5s apart!)."""
    if not (prev and nxt) or dist(prev, nxt) > max_move:
        return None
    return {'cx': prev['cx'] + (nxt['cx'] - prev['cx']) * frac,
            'cy': prev['cy'] + (nxt['cy'] - prev['cy']) * frac,
            'w': (prev['w'] + nxt['w']) / 2,
            'h': (prev['h'] + nxt['h']) / 2}
