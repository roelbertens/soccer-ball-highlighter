# Soccer Ball Highlighter

A Chrome extension that draws a bright ring around the ball in a live football
stream, so people who have trouble spotting the ball (low vision, slow visual
search) can follow the game. Built for my father. Runs entirely in the browser
with TensorFlow.js — no server, no internet needed during playback — and is
light enough for a Chromebook.

## Why

On a wide broadcast shot the ball is a handful of pixels moving fast between
twenty players. If finding it takes you seconds instead of milliseconds, you
watch the replay instead of the goal. A persistent, high-contrast ring fixes
that. Off-the-shelf detectors do poorly here (tiny object, motion blur,
occlusion), which is why this repo is mostly a *data and evaluation* pipeline
around a small YOLOv8 model, not just the extension.

## How it works

Detector + tracker. A fine-tuned YOLOv8 model (single class: `ball`) runs every
~60 ms on a downscaled canvas of the video element; between detections a cheap
motion/brightness tracker keeps the ring glued to the ball, and the ring
"coasts" along the last trajectory for ~180 ms when detection drops out, so it
doesn't flicker. If everything fails, press **L** and click the ball — manual
lock, always works. Details and keyboard shortcuts: `extension/README.md`.

The model ships as fixed-size TensorFlow.js GraphModels at three input
resolutions (512 / 800 / 1280), selectable at runtime: YOLO heads don't convert
with dynamic input shapes, so it's one model per resolution.

## Repository layout

```
extension/            the Chrome extension (MV3): manifest, content.js, bundled tf.min.js
  models/             tfjs models (not in git — see Releases, or build below)
training/             everything to (re)build the model
  train.py            fine-tune YOLOv8 on the ball dataset
  build_size.sh       .pt -> ONNX -> SavedModel -> tfjs GraphModel, fixed size
  annotator.py        local web tool to create/correct labels frame by frame
  find_errors.py      collect model-vs-label disagreements for review (legacy)
  pipeline/           automated data-centric loop:
    extract_frames.py   video -> frames
    autolabel.py        expensive offline ensemble labels unlabeled frames
    refine_labels.py    audits existing labels (auto size-fix, review queue)
    rebuild_review.py   rebuild a review queue from label backups
    build_dataset.py    dataset builder (leave-one-match-out val split)
    eval_model.py       task metrics + leaderboard (not just mAP)
    run_experiments.py  train/eval the setups in experiments.yaml
    run_pipeline.sh     the whole loop in one command
  data/labels/        YOLO labels per match (the hand-checked ground truth)
```

## The data-centric loop

Most of the quality here comes from labels, not architecture. The loop:

1. `extract_frames.py` — sample frames from your own match recordings.
2. `autolabel.py` — an expensive offline ensemble (fine-tuned model +
   YOLOv8x, multi-scale, tiled fallback for tiny balls, temporal-consistency
   gating) writes high-confidence labels and queues doubtful frames.
3. `annotator.py` — review the queues by hand (space = correct, click = ball
   here, x = no ball). Pink box = model prediction, cyan = current label.
4. `refine_labels.py` — audits existing labels against the ensemble: fixes
   box sizes automatically when centers agree, queues disagreements. Always
   backs up first.
5. `build_dataset.py --val-match <match>` — hold out a full match the
   model has never seen; a 20%-tail split leaks stadium/broadcast style.
6. `run_experiments.py` — trains the setups in `experiments.yaml` and ranks
   them in `LEADERBOARD.md` by task metrics measured at the extension's
   realtime settings: detection rate within 24 px, median center error, and
   false-ring rate on no-ball frames (`highlight_score = det_rate − 0.5·frr`).
   mAP is reported but doesn't decide.
7. `build_size.sh` — export the winner, drop it into `extension/models/`.

## Requirements

Python via [uv](https://docs.astral.sh/uv/) — `training/pyproject.toml` defines
everything:

```bash
cd training
uv sync                    # training/labeling/eval env (.venv)
uv sync --extra export     # + the tfjs conversion stack (only for build_size.sh)
source .venv/bin/activate  # then run scripts as plain `python ...`
```

ffmpeg is needed for frame extraction. Training runs fine on Apple Silicon
(`device='mps'`). The extension needs only Chrome (developer mode → load
unpacked → `extension/`).

## Testing

```bash
cd training && python tests/test_pipeline_e2e.py
```

Runs the real pipeline scripts end-to-end in a temp dir against synthetic
frames and a fake `ultralytics` (no GPU, no downloads, ~30 s): label refine,
auto-labeling, review queues, dataset splits, eval/leaderboard, and the
annotator HTTP API — 26 checks.

## Data and models are not in this repo

Training frames come from TV broadcasts and are copyrighted — no video, no
frames, and consequently no trained weights are distributed here. The labels
(plain coordinates) are included. To reproduce: record matches you have the
right to use, run the pipeline above.

## Honest limitations

Wide shots with a sub-10-pixel ball still miss regularly (the coast/tracker
bridges most gaps). DRM-protected streams can't be read by design. Bright
white shirts and shoes are the classic false positives — hence the negative
mining in the annotator (`x` = train as background).

## License and attribution

Code: **AGPL-3.0** (see `LICENSE`). Not a philosophical choice: training and
export are built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
which is AGPL-3.0, and that propagates. See `THIRD_PARTY_NOTICES.md` for all
attributions, including the base model
[`uisikdag/yolo-v8-football-players-detection`](https://huggingface.co/uisikdag/yolo-v8-football-players-detection)
and the bundled TensorFlow.js.
