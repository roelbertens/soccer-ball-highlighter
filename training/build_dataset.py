#!/usr/bin/env python3
# Build a YOLO dataset from the labeled frames (legacy: per-match tail split).
# Only frames WITH a ball label (positives). Per match the last 20%
# (contiguous) becomes validation -> no near-identical frames in both sets.
# Prefer pipeline/build_dataset.py --val-match <match> for an honest
# unseen-match validation split.
import os, shutil, glob, sys

# matches via argv, otherwise all 4
MATCHES = sys.argv[1:] if len(sys.argv) > 1 else ['mex-ecu', 'ivo-nor', 'fra-zwe', 'dui-par']
FRAMES = 'data/frames'
LABELS = 'data/labels'          # per match: data/labels/<match>/labels/*.txt
OUT = 'data/dataset'
VAL_FRAC = 0.20

for split in ['train', 'val']:
    for kind in ['images', 'labels']:
        os.makedirs(f'{OUT}/{kind}/{split}', exist_ok=True)

tot = {'train': 0, 'val': 0, 'neg': 0}
for m in MATCHES:
    lbldir = f'{LABELS}/{m}/labels'
    if not os.path.isdir(lbldir):
        print(f'!! no labels for {m}, skipping'); continue
    txts = sorted(glob.glob(f'{lbldir}/*.txt'))
    # only non-empty labels (= frame with a ball)
    labeled = [t for t in txts if os.path.getsize(t) > 0]
    n = len(labeled); cut = int(n * (1 - VAL_FRAC))
    for idx, t in enumerate(labeled):
        img_name = os.path.basename(t)[:-4]          # strip '.txt' -> 'name.jpg'
        stem = os.path.splitext(img_name)[0]         # -> 'name'
        img = f'{FRAMES}/{m}/{img_name}'
        if not os.path.exists(img):
            continue
        split = 'train' if idx < cut else 'val'
        shutil.copy(img, f'{OUT}/images/{split}/{stem}.jpg')
        shutil.copy(t,   f'{OUT}/labels/{split}/{stem}.txt')
        tot[split] += 1
    # confirmed negatives (too hard -> no ball): included as background in
    # TRAIN so the model learns to draw NOTHING there. Empty label files.
    negp = f'{LABELS}/{m}/negatives.txt'
    nneg = 0
    if os.path.exists(negp):
        for line in open(negp):
            img_name = line.strip()
            if not img_name:
                continue
            img = f'{FRAMES}/{m}/{img_name}'
            if not os.path.exists(img):
                continue
            stem = os.path.splitext(img_name)[0]
            shutil.copy(img, f'{OUT}/images/train/{stem}.jpg')
            open(f'{OUT}/labels/train/{stem}.txt', 'w').close()   # empty = background
            nneg += 1
    tot['neg'] += nneg
    print(f'{m}: {n} labeled -> train {cut}, val {n-cut}  + {nneg} negatives')

with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(f"path: {os.path.abspath(OUT)}\n")
    f.write("train: images/train\nval: images/val\n")
    f.write("nc: 1\nnames: ['ball']\n")

print(f"\nDataset ready: train={tot['train']} (+{tot['neg']} negatives)  val={tot['val']}  -> {OUT}/data.yaml")
