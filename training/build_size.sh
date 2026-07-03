#!/bin/bash
# build_size.sh <imgsz> <weights.pt>
# Build a fixed-size TFJS GraphModel in model_<imgsz>/ from a YOLOv8 .pt:
#   .pt -> ONNX (fixed input) -> TF SavedModel (onnx2tf) -> TFJS GraphModel
# Then copy model_<imgsz>/ to ../extension/models/<imgsz>/
set -e
SZ=$1
PT=${2:-soccer_ball.pt}
cd "$(dirname "$0")"

echo "=== [$SZ] 1) $PT -> ONNX (fixed) ==="
python -c "from ultralytics import YOLO; YOLO('$PT').export(format='onnx', imgsz=$SZ, simplify=True)"
ONNX="${PT%.pt}.onnx"
mv -f "$ONNX" sb_$SZ.onnx

echo "=== [$SZ] 2) ONNX -> TF SavedModel ==="
rm -rf sm_$SZ
onnx2tf -i sb_$SZ.onnx -o sm_$SZ -b 1 > onnx2tf_$SZ.log 2>&1
echo "onnx2tf done"

echo "=== [$SZ] 3) SavedModel -> TFJS ==="
rm -rf model_$SZ
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model sm_$SZ model_$SZ
echo "=== [$SZ] result ==="
ls -la model_$SZ
