#!/usr/bin/env python3
"""Rebuild a lost review queue WITHOUT re-running inference.

Logic: refine_labels --trust-model rewrote every label it agreed with
(SIZEFIX) and queued the rest. So labeled frames whose file is UNCHANGED
vs the most recent backup are exactly the queued-for-review ones. We merge
them back into data/review/<match>.json (autolabel items are kept).

Only meant for ivo-nor / fra-zwe / dui-par after the queue-overwrite bug;
do NOT run for mex-ecu (there OK=170 unchanged frames are genuinely fine).

Usage: python pipeline/rebuild_review.py ivo-nor fra-zwe dui-par
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import MATCHES, LABELS, REVIEW, read_label, save_review

ap = argparse.ArgumentParser()
ap.add_argument('matches', nargs='+', choices=MATCHES)
a = ap.parse_args()

for m in a.matches:
    backups = sorted(glob.glob(os.path.join(LABELS, m, 'labels_backup_*')))
    if not backups:
        print(f'!! {m}: no backup found, cannot rebuild'); continue
    bak = backups[-1]
    cur_dir = os.path.join(LABELS, m, 'labels')
    # never overwrite items already in the queue (they carry better
    # reason/pred info than a rebuilt placeholder)
    qp = os.path.join(REVIEW, m + '.json')
    have = set()
    if os.path.exists(qp):
        try:
            have = {it['file'] for it in json.load(open(qp))}
        except Exception:
            pass
    items = []
    for p in sorted(glob.glob(os.path.join(cur_dir, '*.txt'))):
        if os.path.getsize(p) == 0:
            continue
        name = os.path.basename(p)[:-4]              # strip '.txt' -> 'xxx.jpg'
        if name in have:
            continue
        bp = os.path.join(bak, os.path.basename(p))
        if os.path.exists(bp) and open(bp).read() == open(p).read():
            items.append({'file': name, 'reason': 'loc', 'pred': None,
                          'lab': read_label(m, name)})
    out = save_review(m, items)                       # merges with autolabel items
    print(f'{m}: {len(items)} review items rebuilt (vs {os.path.basename(bak)}) -> {out}')
