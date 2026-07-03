"""Fake `ultralytics` module for offline pipeline tests - no torch, no downloads.

The test harness copies this file to <workdir>/pipeline/ultralytics.py; the
pipeline scripts do sys.path.insert(0, <pipeline dir>), which shadows the real
package. Detections are driven by two JSON files pointed to by env vars:
  BH_FAKE_GT        '<match>/<frame>.jpg' -> [cx, cy, w, h] or null
  BH_FAKE_SPECIALS  '<match>/<frame>.jpg' -> 'faroff' | 'none' | 'lowconf'
"""
import json, os

_gt = None
_sp = None


def _load():
    global _gt, _sp
    if _gt is None:
        _gt = json.load(open(os.environ['BH_FAKE_GT']))
        try:
            _sp = json.load(open(os.environ['BH_FAKE_SPECIALS']))
        except Exception:
            _sp = {}
    return _gt, _sp


class _Box:
    def __init__(self, x, conf):
        self.cls = 0
        self.conf = conf
        self._x = x
        self.xywhn = [self]

    def tolist(self):
        return list(self._x)


class _Conf(list):
    def argmax(self):
        return max(range(len(self)), key=self.__getitem__)


class _Boxes:
    def __init__(self, boxes):
        self._boxes = boxes
        self.conf = _Conf(b.conf for b in boxes)
        self.xywhn = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Result:
    def __init__(self, boxes):
        self.boxes = _Boxes(boxes)
        self.orig_shape = (1078, 1920)


class YOLO:
    def __init__(self, w, **kw):
        self.w = os.path.basename(str(w))
        self.names = {0: 'ball'}

    def load(self, w):
        return self

    def predict(self, source, **kw):
        if not isinstance(source, str):
            return [_Result([])]              # tiled numpy crops: nothing found
        gt, sp = _load()
        key = os.path.join(os.path.basename(os.path.dirname(source)),
                           os.path.basename(source))
        if key not in gt:
            # dataset copies live in images/train|val/ -> match by basename
            bn = os.path.basename(source)
            hits = [k for k in gt if os.path.basename(k) == bn]
            if hits:
                key = hits[0]
        box = gt.get(key)
        mode = sp.get(key)
        if mode == 'none' or box is None:
            return [_Result([])]
        cx, cy, w, h = box
        conf = 0.9
        if mode == 'faroff':
            cx, cy, conf = 0.9, 0.1, 0.99
        elif mode == 'lowconf':
            conf = 0.15
        if self.w == 'best_mex.pt':           # 2nd ensemble member: tiny jitter
            cx += 0.0007
            cy -= 0.0005
        if conf < kw.get('conf', 0.0):
            return [_Result([])]
        return [_Result([_Box([cx, cy, w, h], conf)])]

    def val(self, **kw):
        raise RuntimeError('stub: no real validation in fake ultralytics')

    def train(self, **kw):
        raise RuntimeError('stub: no training in fake ultralytics')
