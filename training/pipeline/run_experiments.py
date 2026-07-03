#!/usr/bin/env python3
"""Train + evaluate all experiments in experiments.yaml, building LEADERBOARD.md.

Resumable: an experiment whose runs/<name>/weights/best.pt exists is not
retrained, only (re)evaluated. So you can Ctrl+C anytime.

  caffeinate -i python pipeline/run_experiments.py
  caffeinate -i python pipeline/run_experiments.py --only e4_v8n_p2_1280 e2_v8n_1280
"""
import argparse, os, sys, yaml, json, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import ROOT
import eval_model as ev

ap = argparse.ArgumentParser()
ap.add_argument('--config', default=os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'experiments.yaml'))
ap.add_argument('--only', nargs='+', default=None)
ap.add_argument('--eval-match', default=None,
                help='additionally eval each model on this held-out match')
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
dfl = cfg['defaults']
os.chdir(ROOT)

from ultralytics import YOLO

for ex in cfg['experiments']:
    if a.only and ex['name'] not in a.only:
        continue
    name = ex['name']
    best = os.path.join(ROOT, 'runs', name, 'weights', 'best.pt')
    if os.path.exists(best):
        print(f'== {name}: best.pt exists, skipping training')
    else:
        print(f'== {name}: training ({ex["base"]} @ {ex["imgsz"]})')
        base = ex['base']
        if '+' in base:                      # 'arch.yaml+init.pt'
            arch, init = base.split('+')
            model = YOLO(arch).load(init)
        else:
            model = YOLO(base)
        model.train(data=dfl['data'], epochs=ex.get('epochs', dfl['epochs']),
                    imgsz=ex['imgsz'], batch=ex['batch'],
                    device=dfl['device'], workers=dfl['workers'],
                    patience=dfl['patience'],
                    project=os.path.join(ROOT, 'runs'), name=name,
                    exist_ok=True, **dfl.get('hyp', {}))
        if not os.path.exists(best):
            print(f'!! {name}: training produced no best.pt, skipping eval')
            continue

    # evaluate at realtime-ish settings
    for match in ([None] + ([a.eval_match] if a.eval_match else [])):
        model = YOLO(best)
        pairs = ev.collect_pairs(match=match)
        if not pairs:
            continue
        imgsz = dfl.get('eval_imgsz', 960)
        m = ev.eval_frames(model, pairs, imgsz, 0.10, dfl['device'], 24.0)
        if not match:
            try:
                v = model.val(data=os.path.join(ROOT, dfl['data']),
                              imgsz=imgsz, device=dfl['device'], verbose=False)
                m['mAP50'] = round(float(v.box.map50), 4)
                m['mAP50_95'] = round(float(v.box.map), 4)
            except Exception as e:
                print('  (val failed:', str(e)[:80], ')')
        ev.update_leaderboard({
            'name': name + (f'@{match}' if match else ''),
            'weights': best, 'imgsz': imgsz, 'conf': 0.10,
            'eval_on': match or 'val',
            'date': datetime.datetime.now().isoformat(), 'metrics': m})
        print(f'   {name} ({match or "val"}): {json.dumps(m)}')

print('\nDone. See LEADERBOARD.md. Export the winner with: '
      'bash build_size.sh 512 runs/<name>/weights/best.pt  (and 800, 1280)')
