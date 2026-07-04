# Evaluation & experiment recipes

How a model earns its place in the extension — the deciding metric, the exact
settings behind every score, and what each training recipe actually does.

The short version: **mAP does not decide.** A model is judged on what a viewer
experiences — is the ring on the ball, and does it stay off everything that
isn't — measured on a **match the model has never seen**.

---

## 1. The deciding metric

```
highlight_score = det_rate − 0.5 · false_ring_rate       (higher = better)
```

Two failures a viewer notices, traded by how much they hurt:

- A **miss** (no ring for a frame) is half-forgiven, because the tracker coasts
  the ring along its last trajectory for ~180 ms — short dropouts barely show.
- A **false ring** (a ring on a white shirt or a boot when there's no ball) has
  no such cover, so it carries the penalty. The 0.5 weight is deliberate: the
  two errors are not symmetric.

`highlight_score` is the sort key for `LEADERBOARD.md`.

## 2. Metrics in full

| metric | definition | better | what it means for the viewer |
|---|---|---|---|
| **highlight_score** | `det_rate − 0.5·false_ring_rate` | ↑ | The verdict. Balances finding the ball against wrongly ringing non-balls. |
| **det_rate** | hits ÷ ball-frames, where a *hit* = predicted center within **24 px** of truth | ↑ | How often the ring lands on the ball. |
| **false_ring_rate** | false rings ÷ no-ball frames | ↓ | How often a ring appears when there's no ball. The distracting failure. |
| **miss_rate** | ball-frames with no detection ÷ ball-frames | ↓ | How often the ring blinks off entirely (bridged live by the coast/tracker). |
| **center_err_px** | median pixel distance prediction → true center (mean also kept) | ↓ | Ring-placement quality. Median so one wild miss doesn't dominate; a large mean-vs-median gap flags rare big errors. |
| **mAP50 / mAP50-95** | standard detection average precision | ↑ | **Context only.** Rewards box-IoU precision the ring never needs, and here it moves opposite to `highlight_score`. Reported, never ranked on. |

The 24 px hit radius is generous on purpose — the ring only has to *sit on* the
ball, not trace its outline.

## 3. Evaluation settings

Evaluation mirrors the live extension, so a leaderboard row predicts field
behaviour, not lab behaviour.

| setting | value | why |
|---|---|---|
| eval resolutions | **800 · 960 · 1280** | One row per size — the exact options the panel offers, so each model is judged at the resolution it will deploy at. |
| confidence threshold | **0.10** | The same threshold the extension runs at. Not tuned per model. |
| hit radius | **24 px** | A prediction is a hit if its center is within 24 px of the true center. |
| validation | **leave-one-match-out** | A whole unseen match is held out (currently `dui-par`). A within-match split leaks stadium/kit/broadcast style into validation and flatters the score. |
| detector | **single model, single pass** | Not the offline labeling ensemble — exactly what runs in the browser. |
| held-out frames | **474 ball + 163 no-ball** | Ball frames drive `det_rate`/`center_err`; confirmed no-ball frames drive `false_ring_rate`. |
| mAP pass | **once, @ 960** | The Ultralytics `val()` pass runs at one resolution so multi-res eval stays cheap (~1–2 min per resolution). |
| device | Apple MPS | Metrics are hardware-independent; device only sets eval wall-time. |

Runs via `pipeline/eval_model.py` (standalone) and inside
`pipeline/run_experiments.py` (per experiment). Both write `runs/leaderboard.json`
and `LEADERBOARD.md`, keyed by `(model, eval_on, imgsz)` so one model has a row
per resolution.

## 4. The models tested

Same data every time; what varies is the **starting weights, model size, and
training resolution**. Each recipe answers one question.

| id | recipe | starts from | train imgsz | role | the question |
|---|---|---|---|---|---|
| **e1** | `players_1280` | `players.pt` | 1280 | reference | The current recipe — the row everything else is measured against. |
| **e2** | `v8n_1280` | `yolov8n.pt` (COCO) | 1280 | probe | Is the football-players model a better start than generic COCO nano? |
| **e3** | `v8s_960` | `yolov8s.pt` | 960 | probe | Does more model capacity help more than more resolution? |
| **e4** | `v8n_p2_1280` | nano + **P2 head** | 1280 | **main bet** | Does a tiny-object detection layer find the far-away, few-pixel ball? |
| **e5** | `v8n_960` | `yolov8n.pt` (COCO) | 960 | **deployed** | Can we trade resolution for speed and keep accuracy? The Chromebook recipe. |

Two **e5** variants sit on the board: `e5 (old)` trained on the pre-audit labels
(the currently shipped model) and `e5 (new)` — the same recipe retrained on the
refined labels. Evaluated on the **same** held-out labels, they isolate whether
better labels alone move the score.

---

## 5. Recipe details — what each run actually does

Defined in [`pipeline/experiments.yaml`](pipeline/experiments.yaml). Every
experiment shares one training config and overrides only `base`, `imgsz`, and
`batch`.

### Shared training config (`defaults`)

| knob | value | note |
|---|---|---|
| epochs | 80 | hard cap |
| patience | 20 | early-stop if val mAP doesn't improve for 20 epochs |
| optimizer | `auto` | Ultralytics picks it — resolves to **AdamW, lr ≈ 0.002, momentum 0.9** |
| data | `data/dataset/data.yaml` | single class: `ball` |
| device / workers | `mps` / 4 | Apple Silicon |
| eval_imgsz | `[800, 960, 1280]` | the eval + deploy resolutions above |

### Augmentations (and why, for a small fast ball)

| aug | value | what it does | why it helps here |
|---|---|---|---|
| `mosaic` | 1.0 | stitches 4 images into one training image | packs many scales/contexts per step — strong for small objects |
| `close_mosaic` | 15 | turns mosaic **off** for the last 15 epochs | final epochs train on natural full frames, matching inference |
| `scale` | 0.5 | random resize ±50% | the ball appears at wildly different sizes (close-up ↔ wide shot) |
| `translate` | 0.1 | shift up to 10% | the ball can be anywhere in frame |
| `fliplr` | 0.5 | horizontal flip, 50% | left/right pitch symmetry; free 2× data |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | hue / saturation / value jitter | broadcast colour, lighting and exposure vary a lot |
| `degrees` | 0.0 | **no rotation** | broadcast framing is upright; rotation distorts the scene without being realistic. Shear/perspective are likewise left off. |

### `base` — two ways to start

- **`players.pt` / `yolov8n.pt` / `yolov8s.pt`** → plain fine-tune from those
  weights.
- **`yolov8-p2.yaml+yolov8n.pt`** (e4) → build the architecture from
  `yolov8-p2.yaml`, then load the pretrained `yolov8n.pt` weights into every
  layer that matches. New layers (the P2 head) start fresh; the rest inherits
  COCO features.

### Per-recipe breakdown

**e1 · `players_1280`** — fine-tune from `players.pt` (a YOLOv8 trained on real
broadcast football: ball/goalkeeper/player/referee) at 1280 px, batch 6. This is
the *reference*: it reproduces the recipe the shipped model came from, so every
other row is a comparison against it.

**e2 · `v8n_1280`** — same size and resolution as e1 but starting from **generic
COCO** `yolov8n.pt` instead of the football model. Isolates one variable: does
the football-specific pretraining actually help, or would plain COCO nano get
there anyway?

**e3 · `v8s_960`** — YOLOv8 **small** (≈3× the parameters of nano) at 960 px,
batch 8. Tests the capacity-vs-resolution trade: is a bigger model at moderate
resolution better than a small model at high resolution? Small is heavier at
inference, so it only earns a deploy slot if the accuracy gain is real.

**e4 · `v8n_p2_1280`** — the main bet. Nano backbone with an added **P2
detection head**, 1280 px, batch 4 (small because the extra high-res feature map
is memory-hungry). *What P2 means:* stock YOLOv8 detects at strides **P3/P4/P5**
— feature maps downsampled 8/16/32×. A ball only a few pixels wide can vanish by
P3. **P2 adds a stride-4 head** on a much higher-resolution feature map, giving
tiny objects far more cells to be detected in. Cost: more compute and memory at
that layer (hence 1280 + batch 4). If better labels help anywhere, it should be
here, on the wide-shot ball.

**e5 · `v8n_960`** — YOLOv8 nano from COCO at 960 px, batch 12. The smallest,
fastest recipe, chosen deliberately for the Chromebook. This is the **currently
deployed** model. Because the ring/coast bridges the occasional miss, high frame
rate on weak hardware often beats sharper-but-slower detection — which is the
whole reason this recipe exists.

---

## 6. Reading the leaderboard / reproducing

```bash
# rebuild the dataset holding out a full match as validation
python pipeline/build_dataset.py --val-match dui-par

# train + evaluate every recipe at 800/960/1280 (resumable)
python pipeline/run_experiments.py --eval-match dui-par

# or evaluate one set of weights on the held-out match at all resolutions
python pipeline/eval_model.py --weights runs/<name>/weights/best.pt --match dui-par
```

Rows sort by `highlight_score`. Read deltas with care: a whole held-out match is
one noisy sample, so a swing of ≈0.03 is within run-to-run noise. And better
*labels* don't always raise `det_rate` — much label work (box-size fixes, mined
negatives) improves mAP and false-ring behaviour while `det_rate` only tracks
whether the ball's *center* is found. **Judge a change by the metric it was meant
to move.**
