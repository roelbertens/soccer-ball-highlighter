#!/usr/bin/env python3
"""Dataset builder. Default = per-match tail split (last 20% as val).
Better: --val-match holds out a FULL match as validation. The tail split
leaks broadcast style/stadium into val; leave-one-match-out tells you how
the model does on a match it has never seen (what the extension actually
faces).

Usage:
  python pipeline/build_dataset.py --val-match dui-par
  python pipeline/build_dataset.py                    # tail split
"""
import argparse, os, shutil, glob, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import ROOT, MATCHES, LABELS, FRAMES

ap = argparse.ArgumentParser()
ap.add_argument('--matches', nargs='+', default=MATCHES, choices=MATCHES)
ap.add_argument('--val-match', default=None, choices=MATCHES)
ap.add_argument('--val-frac', type=float, default=0.20)
ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'dataset'))
a = ap.parse_args()

OUT = a.out
if os.path.isdir(OUT):
    shutil.rmtree(OUT)          # always rebuild clean (no stale frames)
for split in ['train', 'val']:
    for kind in ['images', 'labels']:
        os.makedirs(f'{OUT}/{kind}/{split}', exist_ok=True)

tot = {'train': 0, 'val': 0, 'neg': 0}
for m in a.matches:
    lbldir = f'{LABELS}/{m}/labels'
    if not os.path.isdir(lbldir):
        print(f'!! no labels for {m}, skipping'); continue
    txts = sorted(glob.glob(f'{lbldir}/*.txt'))
    labeled = [t for t in txts if os.path.getsize(t) > 0]
    n = len(labeled)
    cut = n if (a.val_match and m != a.val_match) else int(n * (1 - a.val_frac))
    if a.val_match == m:
        cut = 0                                  # whole match -> val
    for idx, t in enumerate(labeled):
        img_name = os.path.basename(t)[:-4]      # 'xxx.jpg'
        stem = os.path.splitext(img_name)[0]
        img = f'{FRAMES}/{m}/{img_name}'
        if not os.path.exists(img):
            continue
        split = 'train' if idx < cut else 'val'
        shutil.copy(img, f'{OUT}/images/{split}/{stem}.jpg')
        shutil.copy(t, f'{OUT}/labels/{split}/{stem}.txt')
        tot[split] += 1
    # negatives as background: into train, except for the held-out match ->
    # into val, so false_ring_rate is measurable on the val split too
    negp = f'{LABELS}/{m}/negatives.txt'
    nneg = 0
    nsplit = 'val' if m == a.val_match else 'train'
    if os.path.exists(negp):
        for line in open(negp):
            img_name = line.strip()
            if not img_name:
                continue
            img = f'{FRAMES}/{m}/{img_name}'
            if not os.path.exists(img):
                continue
            stem = os.path.splitext(img_name)[0]
            shutil.copy(img, f'{OUT}/images/{nsplit}/{stem}.jpg')
            open(f'{OUT}/labels/{nsplit}/{stem}.txt', 'w').close()
            nneg += 1
    tot['neg'] += nneg
    print(f'{m}: {n} labeled -> train {cut}, val {n - cut}  + {nneg} negatives')

with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(f"path: {os.path.abspath(OUT)}\n")
    f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['ball']\n")
print(f"\nDataset: train={tot['train']} (+{tot['neg']} neg)  val={tot['val']}"
      f"  val-match={a.val_match or f'tail {a.val_frac:.0%} per match'}")
