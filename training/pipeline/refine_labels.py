#!/usr/bin/env python3
"""Audit + improve EXISTING labels (data-centric step for mex-ecu and later matches).

What it does per labeled frame:
  1. SIZEFIX (automatic, safe): manual clicks all have the fixed 24px box
     (w=0.0125, h=0.0223) - wrong for close-ups where the ball is 5x bigger.
     If the ensemble agrees on the CENTER (within --agree), the box SIZE is
     replaced by the model's tight box. Center is kept from the human click
     (blended 70/30 toward the model when very close).
  2. LOC: strong model detection far away from the label -> review queue.
  3. JUMP: label is a temporal outlier vs its neighbor labels -> review queue.
  4. MISS: ensemble finds nothing near a label -> low-priority review item.

A full backup of the label dir is made first (labels_backup_<stamp>/).

Usage:
  caffeinate -i python pipeline/refine_labels.py mex-ecu
  python pipeline/refine_labels.py mex-ecu --dry-run     # report only
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import (MATCHES, BallDetector, list_frames, read_label,
                       write_label, save_review, backup_labels, is_fixed_size,
                       temporal_flag, dist, frames_dir)

ap = argparse.ArgumentParser()
ap.add_argument('matches', nargs='+', choices=MATCHES)
ap.add_argument('--agree', type=float, default=0.015,
                help='center distance below which model and label "agree"')
ap.add_argument('--far', type=float, default=0.03,
                help='center distance above which a strong det disagrees')
ap.add_argument('--strong', type=float, default=0.40)
ap.add_argument('--device', default='mps')
ap.add_argument('--imgsz', type=int, nargs='+', default=[1280, 1920])
ap.add_argument('--weights', nargs='+', default=None)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--dry-run', action='store_true')
ap.add_argument('--trust-model', action='store_true',
                help='labels came from a weak model (not human clicks): on '
                     'agreement take the ensemble center fully, not 70/30')
a = ap.parse_args()

det = BallDetector(weights=a.weights, imgszs=a.imgsz, device=a.device)

for m in a.matches:
    frames = [f for f in list_frames(m) if read_label(m, f)]
    if a.limit:
        frames = frames[:a.limit]
    if not frames:
        print(f'!! {m}: no labeled frames'); continue
    if not a.dry_run:
        b = backup_labels(m)
        print(f'== {m}: {len(frames)} labeled frames (backup: {b})')
    else:
        print(f'== {m}: {len(frames)} labeled frames (DRY RUN)')

    labs = {f: read_label(m, f) for f in frames}
    review, n_sizefix, n_ok = [], 0, 0

    for i, f in enumerate(frames, 1):
        if i % 25 == 0:
            print(f'   ...{i}/{len(frames)}  (sizefix {n_sizefix}, review {len(review)})',
                  flush=True)
        lab = labs[f]
        ds = det.detect(os.path.join(frames_dir(m), f))
        near = [d for d in ds if dist(d, lab) < a.agree]
        strong_far = [d for d in ds if d['conf'] >= a.strong and dist(d, lab) > a.far]

        if near:
            d = max(near, key=lambda x: x['conf'])
            if is_fixed_size(lab) or abs(lab['w'] - d['w']) > 0.5 * lab['w'] or a.trust_model:
                k = 0.0 if a.trust_model else 0.7   # weight of the old center
                new = {'cx': k * lab['cx'] + (1 - k) * d['cx'],
                       'cy': k * lab['cy'] + (1 - k) * d['cy'],
                       'w': d['w'], 'h': d['h']}
                if not a.dry_run:
                    write_label(m, f, new)
                labs[f] = new
                n_sizefix += 1
            else:
                n_ok += 1
        elif strong_far:
            d = max(strong_far, key=lambda x: x['conf'])
            review.append({'file': f, 'reason': 'loc', 'pred': d, 'lab': lab})
        elif ds:   # found something, but in the ambiguous 0.015-0.03 zone / weak
            review.append({'file': f, 'reason': 'loc', 'pred': ds[0], 'lab': lab})
        else:
            review.append({'file': f, 'reason': 'miss', 'pred': None, 'lab': lab})

    # temporal outliers among (possibly updated) labels
    seq = [labs.get(f) for f in frames]
    flagged = {it['file'] for it in review}
    for i, f in enumerate(frames):
        if f in flagged:
            continue
        prev = seq[i - 1] if i > 0 else None
        nxt = seq[i + 1] if i + 1 < len(frames) else None
        if temporal_flag(prev, seq[i], nxt):
            review.append({'file': f, 'reason': 'loc', 'pred': None, 'lab': seq[i]})

    order = {'fp': 0, 'loc': 1, 'miss': 2}   # same order as bh_common.save_review
    review.sort(key=lambda it: order.get(it['reason'], 3))
    p = save_review(m, review)
    print(f'{m}: OK {n_ok}  SIZEFIX {n_sizefix}  REVIEW {len(review)} -> {p}')
