#!/bin/bash
# Full data-centric improvement loop. Run from ball-highlighter-build/:
#   bash pipeline/run_pipeline.sh            # all stages
#   bash pipeline/run_pipeline.sh labels     # only stage 1-3 (label work)
#   bash pipeline/run_pipeline.sh train      # only stage 4-6 (dataset/train/eval)
# Everything long-running is wrapped in caffeinate so the Mac stays awake.
set -e
cd "$(dirname "$0")/.."
STAGE="${1:-all}"

python -c 'import ultralytics' 2>/dev/null || {
  echo "!! ultralytics not found - activate the env first:"
  echo "     uv sync && source .venv/bin/activate     (or: conda activate balexport)"
  exit 1
}

if [[ "$STAGE" == "all" || "$STAGE" == "labels" ]]; then
  echo "== 1. extract frames for the 3 unlabeled matches (skips existing) =="
  caffeinate -i python pipeline/extract_frames.py ivo-nor fra-zwe dui-par --fps 2

  echo "== 2. audit + auto-improve existing labels (backups made) =="
  caffeinate -i python pipeline/refine_labels.py mex-ecu
  caffeinate -i python pipeline/refine_labels.py ivo-nor fra-zwe dui-par --trust-model

  echo "== 3. re-detect frames the earlier weak-model run left empty =="
  caffeinate -i python pipeline/autolabel.py ivo-nor fra-zwe dui-par --redo-empty

  echo ""
  echo ">>> NOW: python annotator.py  -> review the queues (data/review/*.json)"
  echo ">>> then rerun: bash pipeline/run_pipeline.sh train"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  echo "== 4. build dataset (hold out dui-par as unseen-match validation) =="
  python pipeline/build_dataset.py --val-match dui-par

  echo "== 5. train + evaluate all model setups (resumable) =="
  caffeinate -i python pipeline/run_experiments.py --eval-match dui-par

  echo "== 6. leaderboard =="
  cat LEADERBOARD.md
  echo ""
  echo ">>> Export the winner for the extension:"
  echo ">>>   bash build_size.sh 512 runs/<winner>/weights/best.pt"
  echo ">>>   bash build_size.sh 800 runs/<winner>/weights/best.pt"
  echo ">>>   bash build_size.sh 1280 runs/<winner>/weights/best.pt"
  echo ">>> then copy model_*/ into ../ball-highlighter-v2/models/"
fi
