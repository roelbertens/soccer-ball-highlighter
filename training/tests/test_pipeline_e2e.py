#!/usr/bin/env python3
"""End-to-end smoke test for the labeling/eval pipeline.

Runs the REAL pipeline scripts as subprocesses in a temp directory against
synthetic frames and a fake `ultralytics` (no torch, no downloads, ~30 s).
Covers: refine_labels (sizefix/loc/miss), rebuild_review (merge, reason
preservation), autolabel (auto/review gating, idempotency, human labels
untouched), build_dataset (leave-one-match-out), eval_model (+leaderboard)
and the annotator HTTP API.

Usage:  python tests/test_pipeline_e2e.py        (from training/)
Needs:  opencv-python (already a project dependency).
"""
import json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
PY = sys.executable
RESULTS = []


def check(name, ok, info=''):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{info}]" if info else ''))


def sh(root, env, *args):
    r = subprocess.run([PY] + list(args), cwd=root, capture_output=True,
                       text=True, env=dict(os.environ, **env))
    if r.returncode != 0:
        print('--- stdout ---\n' + r.stdout[-3000:])
        print('--- stderr ---\n' + r.stderr[-3000:])
        raise SystemExit(f'subprocess failed: {args}')
    return r.stdout


def read_lab(root, m, f):
    p = os.path.join(root, 'data/labels', m, 'labels', f + '.txt')
    if not os.path.exists(p):
        return 'absent'
    s = open(p).read().split()
    return None if not s else [float(x) for x in s[1:]]


def main():
    import numpy as np, cv2
    root = tempfile.mkdtemp(prefix='bh_e2e_')
    print('workdir:', root)

    shutil.copytree(os.path.join(TRAINING, 'pipeline'), os.path.join(root, 'pipeline'),
                    ignore=shutil.ignore_patterns('__pycache__', 'ultralytics*'))
    shutil.copy(os.path.join(TRAINING, 'annotator.py'), os.path.join(root, 'annotator.py'))
    shutil.copy(os.path.join(HERE, 'fake_ultralytics.py'),
                os.path.join(root, 'pipeline', 'ultralytics.py'))
    open(os.path.join(root, 'best_mex.pt'), 'w').close()   # -> 2-model ensemble

    # ---------------- synthetic data ----------------
    gt = {}

    def frames(match, n, ball_at):
        d = os.path.join(root, 'data/frames', match)
        os.makedirs(d, exist_ok=True)
        for i in range(1, n + 1):
            f = f'{match}_{i:04d}.jpg'
            b = ball_at(i)
            img = np.zeros((1078, 1920, 3), np.uint8)
            img[:] = (40, 120, 40)
            if b:
                cv2.circle(img, (int(b[0] * 1920), int(b[1] * 1078)), 8,
                           (255, 255, 255), -1)
            cv2.imwrite(os.path.join(d, f), img)
            gt[f'{match}/{f}'] = [b[0], b[1], 0.0083, 0.0148] if b else None

    frames('mex-ecu', 14, lambda i: (0.2 + 0.04 * i, 0.5) if i <= 11 else None)
    # step 0.03/frame: 2-frame neighbor distance 0.06 stays under the 0.08
    # temporal-jump threshold, so outlier detection can work
    frames('dui-par', 10, lambda i: (0.3 + 0.03 * i, 0.6) if i <= 8 else None)
    specials = {'mex-ecu/mex-ecu_0007.jpg': 'faroff',
                'mex-ecu/mex-ecu_0008.jpg': 'none',
                'dui-par/dui-par_0006.jpg': 'lowconf',
                'dui-par/dui-par_0007.jpg': 'faroff'}
    json.dump(gt, open(os.path.join(root, 'gt.json'), 'w'))
    json.dump(specials, open(os.path.join(root, 'specials.json'), 'w'))
    env = {'BH_FAKE_GT': os.path.join(root, 'gt.json'),
           'BH_FAKE_SPECIALS': os.path.join(root, 'specials.json')}

    # ---------------- seed labels ----------------
    def seed(match, i, cx, cy, w='0.012500', h='0.022263', empty=False):
        d = os.path.join(root, 'data/labels', match, 'labels')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'{match}_{i:04d}.jpg.txt'), 'w') as fh:
            if not empty:
                fh.write(f'0 {cx:.6f} {cy:.6f} {w} {h}\n')

    for i in range(1, 7):                                   # fixed-size clicks at GT
        seed('mex-ecu', i, 0.2 + 0.04 * i, 0.5)
    seed('mex-ecu', 7, 0.2 + 0.04 * 7, 0.5)                 # fake goes far  -> loc
    seed('mex-ecu', 8, 0.2 + 0.04 * 8, 0.5, '0.008000', '0.014000')  # fake none -> miss
    seed('mex-ecu', 9, 0.9, 0.9)                            # wrong center   -> loc
    seed('mex-ecu', 10, 0, 0, empty=True)                   # human 'no ball'
    open(os.path.join(root, 'data/labels/mex-ecu/negatives.txt'),
         'w').write('mex-ecu_0012.jpg\n')
    for i in range(1, 4):                                   # dui-par: human labels
        seed('dui-par', i, 0.3 + 0.03 * i, 0.6, '0.008000', '0.014000')
    human_before = open(os.path.join(
        root, 'data/labels/dui-par/labels/dui-par_0001.jpg.txt')).read()

    # ---------------- refine_labels ----------------
    sh(root, env, 'pipeline/refine_labels.py', 'mex-ecu')
    labs = {i: read_lab(root, 'mex-ecu', f'mex-ecu_{i:04d}.jpg') for i in range(1, 11)}
    q = json.load(open(os.path.join(root, 'data/review/mex-ecu.json')))
    reasons = {it['file']: it['reason'] for it in q}
    check('refine: sizefix applied to fixed-size labels',
          all(labs[i] is not None and labs[i] != 'absent'
              and abs(labs[i][2] - 0.0083) < 0.002 for i in range(1, 7)))
    check('refine: backup made',
          any(d.startswith('labels_backup_') for d in
              os.listdir(os.path.join(root, 'data/labels/mex-ecu'))))
    check('refine: far-off model box -> loc', reasons.get('mex-ecu_0007.jpg') == 'loc')
    check('refine: no detection -> miss', reasons.get('mex-ecu_0008.jpg') == 'miss')
    check('refine: wrong label center -> loc', reasons.get('mex-ecu_0009.jpg') == 'loc')
    check('refine: empty (no-ball) label untouched', labs[10] is None)

    # ---------------- rebuild_review ----------------
    sh(root, env, 'pipeline/rebuild_review.py', 'mex-ecu')
    q2 = json.load(open(os.path.join(root, 'data/review/mex-ecu.json')))
    files = [it['file'] for it in q2]
    check('rebuild: no duplicate queue items', len(files) == len(set(files)))
    check('rebuild: existing reasons preserved',
          {it['file']: it['reason'] for it in q2}.get('mex-ecu_0008.jpg') == 'miss')

    # ---------------- autolabel ----------------
    out = sh(root, env, 'pipeline/autolabel.py', 'dui-par', '--no-tiles')
    dq = json.load(open(os.path.join(root, 'data/review/dui-par.json')))
    dr = {it['file']: it['reason'] for it in dq}
    d4 = read_lab(root, 'dui-par', 'dui-par_0004.jpg')
    check('autolabel: AUTO > 0', 'AUTO 0 ' not in out)
    check('autolabel: low-conf queued as fp', dr.get('dui-par_0006.jpg') == 'fp')
    check('autolabel: temporal jump queued as loc', dr.get('dui-par_0007.jpg') == 'loc')
    check('autolabel: human labels untouched',
          open(os.path.join(root, 'data/labels/dui-par/labels/dui-par_0001.jpg.txt')
               ).read() == human_before)
    check('autolabel: wrote labels at GT position',
          d4 not in (None, 'absent') and abs(d4[0] - (0.3 + 0.03 * 4)) < 0.01)
    out2 = sh(root, env, 'pipeline/autolabel.py', 'dui-par', '--no-tiles')
    check('autolabel: idempotent on rerun', 'AUTO 0 ' in out2)

    # ---------------- build_dataset ----------------
    sh(root, env, 'pipeline/build_dataset.py', '--val-match', 'dui-par')
    tr = os.listdir(os.path.join(root, 'data/dataset/images/train'))
    vl = os.listdir(os.path.join(root, 'data/dataset/images/val'))
    check('dataset: held-out split is clean',
          all(f.startswith('mex') for f in tr) and all(f.startswith('dui') for f in vl),
          f'train {len(tr)}, val {len(vl)}')
    check('dataset: negative included as background', os.path.getsize(
        os.path.join(root, 'data/dataset/labels/train/mex-ecu_0012.txt')) == 0)

    # ---------------- eval_model + leaderboard ----------------
    open(os.path.join(root, 'fake.pt'), 'w').close()
    for name in ('smoke1', 'smoke2'):
        sh(root, env, 'pipeline/eval_model.py', '--weights', 'fake.pt',
           '--name', name, '--device', 'cpu', '--no-map')
    lb = json.load(open(os.path.join(root, 'runs/leaderboard.json')))
    check('eval: leaderboard has 2 rows', len(lb) == 2)
    check('eval: det_rate > 0', lb[0]['metrics']['det_rate'] > 0,
          f"det_rate {lb[0]['metrics']['det_rate']}")
    check('eval: LEADERBOARD.md written',
          os.path.exists(os.path.join(root, 'LEADERBOARD.md')))

    # ---------------- annotator HTTP API ----------------
    port = 8123
    proc = subprocess.Popen([PY, 'annotator.py'], cwd=root,
                            env=dict(os.environ, BH_PORT=str(port)),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f'http://127.0.0.1:{port}'
        for _ in range(50):
            try:
                urllib.request.urlopen(base + '/', timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        html = urllib.request.urlopen(base + '/').read().decode()
        check('annotator: serves UI', 'Ball annotation' in html)
        items = json.load(urllib.request.urlopen(base + '/api/items?match=mex-ecu&only=1'))
        check('annotator: lists labeled frames', len(items) >= 9, f'{len(items)} items')
        errs = json.load(urllib.request.urlopen(base + '/api/errors?match=mex-ecu'))
        check('annotator: serves review queue',
              len(errs) >= 3 and errs[0].get('match') == 'mex-ecu')

        def post(payload):
            req = urllib.request.Request(base + '/api/save',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)

        post({'match': 'mex-ecu', 'file': 'mex-ecu_0001.jpg',
              'label': {'cx': 0.5, 'cy': 0.5, 'w': 0.01, 'h': 0.02}, 'neg': False})
        check('annotator: click saves label',
              read_lab(root, 'mex-ecu', 'mex-ecu_0001.jpg') == [0.5, 0.5, 0.01, 0.02])
        post({'match': 'mex-ecu', 'file': 'mex-ecu_0003.jpg', 'label': None, 'neg': True})
        negs = open(os.path.join(root, 'data/labels/mex-ecu/negatives.txt')).read()
        check('annotator: x empties label + records negative',
              read_lab(root, 'mex-ecu', 'mex-ecu_0003.jpg') is None
              and 'mex-ecu_0003.jpg' in negs)
        for bad in ('/img/..%2f..%2fetc%2fpasswd', '/img/noslash'):
            code = 0
            try:
                urllib.request.urlopen(base + bad)
            except urllib.error.HTTPError as e:
                code = e.code
            check(f'annotator: {bad} rejected', code == 404)
    finally:
        proc.terminate()

    shutil.rmtree(root, ignore_errors=True)
    print(f'\n{sum(RESULTS)}/{len(RESULTS)} checks passed')
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == '__main__':
    main()
