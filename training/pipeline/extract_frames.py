#!/usr/bin/env python3
"""Extract frames from videos/<match>.mov -> data/frames/<match>/<match>_NNNN.jpg

Numbering starts at 0001 (same convention as the existing mex-ecu frames).
Refuses to touch a non-empty frames dir unless --force (protects mex-ecu!).

Usage:
  python pipeline/extract_frames.py ivo-nor fra-zwe dui-par --fps 2
"""
import argparse, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bh_common import ROOT, VIDEOS, frames_dir, list_frames, MATCHES

ap = argparse.ArgumentParser()
ap.add_argument('matches', nargs='+', choices=MATCHES)
ap.add_argument('--fps', type=float, default=2.0,
                help='frames per second to sample (mex-ecu looks like ~2/s)')
ap.add_argument('--start', default=None, help='e.g. 00:05:00 skip pre-match')
ap.add_argument('--duration', default=None, help='e.g. 00:06:00 limit length')
ap.add_argument('--force', action='store_true')
a = ap.parse_args()

for m in a.matches:
    vid = os.path.join(VIDEOS, m + '.mov')
    if not os.path.exists(vid):
        print(f'!! {vid} not found, skipping'); continue
    out = frames_dir(m)
    if list_frames(m) and not a.force:
        print(f'!! {m}: frames already exist ({len(list_frames(m))}), skipping '
              f'(use --force to re-extract - this RENUMBERS frames and breaks labels!)')
        continue
    os.makedirs(out, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    if a.start:
        cmd += ['-ss', a.start]
    cmd += ['-i', vid]
    if a.duration:
        cmd += ['-t', a.duration]
    cmd += ['-vf', f'fps={a.fps}', '-q:v', '2',
            os.path.join(out, f'{m}_%04d.jpg')]
    print('->', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print(f'{m}: {len(list_frames(m))} frames')
