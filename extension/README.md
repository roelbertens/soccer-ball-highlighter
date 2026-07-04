# ⚽ Ball Highlighter — Chrome extension (v2, YOLOv8)

Chrome extension that tracks the ball in a football stream and draws a bright
ring around it. Runs entirely in the browser → works on **Chrome OS**. No
internet needed while watching: the model is bundled with the extension.

## Quickstart (Chromebook or desktop Chrome)

1. Get the `extension/` folder onto the machine (USB, Drive, git clone) —
   including `models/` (not in git; build them or download a release).
2. Open Chrome → address bar: `chrome://extensions`
3. Enable **Developer mode** (top right).
4. Click **Load unpacked** → select the `extension/` folder.
5. Open a **football video** (a full match on YouTube is the easiest start).
   Click the video once so it has focus.
6. Wait for "⏳ Loading model…" → then "✅ Ready". The ring appears around the
   ball automatically.

Ring not tracking for a moment? Press **L** and **click the ball** — it locks
on immediately. That's the reliable fallback.

## Controls

| Key | Action |
|------|-------|
| **Alt + B** | show/hide panel |
| **Alt + H** | highlighting on/off |
| **L** | lock mode → then click the ball |

In the panel (Alt + B): tracker on/off, trail, spotlight (dim the rest), ring
size, detection sensitivity, tracker grip, **detection resolution**, ring color.

## Detection resolution

The extension uses a **football-trained YOLOv8 model** (ball + players in the
base data; only the ball is ringed). In the panel you pick the resolution the
detector runs at:

- **800px** — fastest; pick this if the video stutters on a Chromebook (default).
- **960px** — balance between speed and accuracy.
- **1280px** — most accurate for the small, fast ball, but heaviest. Only
  smooth on a fast desktop; probably too slow on a Chromebook.

These are also the resolutions the training pipeline evaluates at, so the
leaderboard predicts what each panel setting will do.

Switching briefly loads a different model (~1 s).

## Tuning during a match

- **Detection sensitivity** higher → finds the ball more often, but more false
  rings. Lower → calmer, but misses the ball sometimes.
- **Tracker grip** lower → lets go sooner when unsure (less latching onto a
  wrong object). Higher → sticks longer.

## Honest caveats

- **DRM streams** don't expose their pixels; you'll get a notice and it won't
  work. Test with a YouTube video or a free live stream.
- The ball remains the hardest object (small, fast, often occluded). On a
  **wide camera shot** the ball is only a few pixels; auto-detection misses
  regularly there (the ball is found in a share of frames, not all). Between
  detections the ring "coasts" along the last direction so it doesn't blink.
  Close-ups work best.
- The **L-lock** (press L, click the ball) is the dependable fallback whenever
  auto-detection loses it.

## Files

```
manifest.json          – extension config (MV3)
content.js             – logic; runDetector() = YOLOv8 block, trackStep() = the tracker
lib/tf.min.js          – TensorFlow.js (patched: no eval, MV3-proof)
models/800|960|1280/   – YOLOv8 models (TensorFlow.js GraphModel per resolution, float16)
```

## Rebuilding the model (developer)

Base model: `players.pt` (YOLOv8, classes ball/goalkeeper/player/referee, from
HuggingFace `uisikdag/yolo-v8-football-players-detection`, trained on real
broadcast footage). Only class 0 = "ball" is used (output channel 4). The tfjs
models are built with the scripts in `../training/`:

```bash
# per resolution: .pt -> ONNX (fixed) -> TF SavedModel (onnx2tf) -> TFJS GraphModel (float16)
bash build_size.sh 800 <weights.pt>    # and 960, 1280
```

Note: `format='tfjs'` in ultralytics was replaced by LiteRT as of 8.4.83, and
YOLO heads don't convert with *dynamic* input — hence a **fixed model per
resolution**. Want a different/better `.pt`? Run the training pipeline in
`../training/pipeline/`, then rebuild; the extension needs no changes (as long
as it's 1 class with output `1×5×N`).
