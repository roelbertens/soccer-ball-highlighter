#!/usr/bin/env python3
"""Auto-label frames with the expensive offline ensemble (multi-model,
multi-scale, tiled fallback) + temporal consistency.

Per frame the outcome is one of:
  AUTO    high confidence + temporally consistent  -> label written to disk
  REVIEW  low conf / jump / interpolated / conflict -> queued in data/review/<match>.json
  NONE    nothing found, no neighbor support        -> left unlabeled

Existing label files (incl. empty = reviewed 'no ball') are human ground
truth and are never overwritten (use refine_labels.py to audit those).

Usage:
  caffeinate -i python pipeline/autolabel.py ivo-nor fra-zwe dui-par
Options of interest: --auto-conf 0.5  --review-conf 0.12  --device mps
"""
import argparse, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import (MATCHES, BallDetector, list_frames, read_label,
                       has_label_file, write_label, save_review, read_negatives,
                       temporal_flag, interpolate, dist, frames_dir)

ap = argparse.ArgumentParser()
ap.add_argument('matches', nargs='+', choices=MATCHES)
ap.add_argument('--auto-conf', type=float, default=0.50)
ap.add_argument('--review-conf', type=float, default=0.12)
ap.add_argument('--jump', type=float, default=0.08,
                help='normalized center jump vs both neighbors => review')
ap.add_argument('--device', default='mps')
ap.add_argument('--imgsz', type=int, nargs='+', default=[1280, 1920])
ap.add_argument('--weights', nargs='+', default=None)
ap.add_argument('--no-tiles', action='store_true')
ap.add_argument('--limit', type=int, default=0, help='only first N frames (smoke test)')
ap.add_argument('--redo-empty', action='store_true',
                help='re-detect frames whose label file is EMPTY (leftovers of an '
                     'earlier weak-model run); negatives.txt frames stay untouched')
a = ap.parse_args()

det = BallDetector(weights=a.weights, imgszs=a.imgsz, device=a.device,
                   use_tiles=not a.no_tiles)

for m in a.matches:
    frames = list_frames(m)
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        print(f'!! {m}: no frames - run extract_frames.py first'); continue
    negs = read_negatives(m)
    print(f'== {m}: {len(frames)} frames')

    # pass 1: detect
    best = {}      # frame -> best detection (or None)
    human = {}     # frame -> existing human/previous label
    for i, f in enumerate(frames, 1):
        if i % 25 == 0:
            print(f'   ...{i}/{len(frames)}', flush=True)
        lab = read_label(m, f)
        if lab:
            human[f] = lab
            best[f] = None
            continue
        if f in negs or (has_label_file(m, f) and not a.redo_empty):
            best[f] = None      # empty label file = reviewed 'no ball' -> skip
            continue
        ds = det.detect(os.path.join(frames_dir(m), f))
        best[f] = ds[0] if ds and ds[0]['conf'] >= a.review_conf else None

    # pass 2: temporal decisions
    def pos(f):
        return human.get(f) or best.get(f)

    review, n_auto, n_none = [], 0, 0
    for i, f in enumerate(frames):
        if f in human or f in negs or (has_label_file(m, f) and not a.redo_empty):
            continue
        prev = pos(frames[i - 1]) if i > 0 else None
        nxt = pos(frames[i + 1]) if i + 1 < len(frames) else None
        d = best.get(f)
        if d:
            jumped = temporal_flag(prev, d, nxt, jump=a.jump)
            supported = ((prev and dist(prev, d) < a.jump)
                         or (nxt and dist(nxt, d) < a.jump))
            if d['conf'] >= a.auto_conf and not jumped:
                write_label(m, f, d); n_auto += 1
            elif d['conf'] >= 0.25 and supported and not jumped:
                write_label(m, f, d); n_auto += 1   # medium conf but trajectory agrees
            else:
                review.append({'file': f, 'reason': 'loc' if jumped else 'fp',
                               'pred': d, 'lab': None})
        else:
            guess = interpolate(prev, nxt)
            if guess:
                guess['conf'] = 0.0
                review.append({'file': f, 'reason': 'miss', 'pred': guess, 'lab': None})
            else:
                n_none += 1

    p = save_review(m, review)
    print(f'{m}: AUTO {n_auto}  REVIEW {len(review)} -> {p}  NONE {n_none}  '
          f'(human labels untouched: {len(human)})')
print('\nNext: python annotator.py -> red review buttons to work through the queues.')
