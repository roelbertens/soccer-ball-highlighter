# Third-party notices

## Ultralytics YOLOv8
- https://github.com/ultralytics/ultralytics — License: **AGPL-3.0**
- Used for training, validation and model export (`training/`). The detection
  models used by the extension are YOLOv8 architectures fine-tuned with this
  library. AGPL-3.0 is the reason this repository is AGPL-3.0.

## Base model: uisikdag/yolo-v8-football-players-detection
- https://huggingface.co/uisikdag/yolo-v8-football-players-detection
- YOLOv8 model trained on broadcast football footage (classes: ball,
  goalkeeper, player, referee). Our ball model was fine-tuned starting from
  these weights (`players.pt`). The model card publishes no explicit license;
  the weights are used locally as a fine-tuning starting point only and are
  **not** redistributed in this repository.

## Ultralytics pretrained weights (yolov8n/s/x.pt)
- https://github.com/ultralytics/assets — License: AGPL-3.0
- `yolov8x.pt` is used offline as a second opinion in the auto-labeling
  ensemble (`training/pipeline/autolabel.py`); `yolov8n/s.pt` as training
  starting points in the experiments. Not redistributed.

## TensorFlow.js
- https://github.com/tensorflow/tfjs — License: **Apache-2.0**
- Bundled as `extension/lib/tf.min.js` (minified build, lightly patched to
  avoid `eval` for Chrome Manifest V3 compliance). The Apache-2.0 license and
  notices of TensorFlow.js apply to that file; a copy of the license is
  included as `extension/lib/LICENSE-tensorflowjs.txt`.

## Build-time tools (not bundled or redistributed)
- onnx2tf — MIT — https://github.com/PINTO0309/onnx2tf
- ONNX — Apache-2.0 — https://github.com/onnx/onnx
- tensorflowjs converter — Apache-2.0
- ffmpeg — LGPL/GPL — frame extraction

## Training footage
- Frames were sampled from television broadcasts for private, non-commercial
  model training. No video, frames, or imagery are distributed with this
  repository; only label coordinates created by the author.
