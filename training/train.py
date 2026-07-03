#!/usr/bin/env python3
# Fine-tune YOLOv8 on the ball class, starting from the current model.
# (For systematic comparisons use pipeline/run_experiments.py instead.)
import os
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
model = YOLO('players.pt')          # start from the broadcast-trained base model
model.train(
    data=os.path.join(HERE, 'data/dataset/data.yaml'),
    epochs=80,
    imgsz=1280,                     # high -> the small ball gets pixels
    batch=6,
    device='mps',                   # Apple Silicon GPU
    workers=4,
    patience=20,
    project=os.path.join(HERE, 'runs'), name='ball_ft',
    # augmentations (help the small, fast ball)
    mosaic=1.0, close_mosaic=15,
    scale=0.5, translate=0.1, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, shear=0.0, perspective=0.0,
)
print("Done. Best weights: runs/ball_ft/weights/best.pt")
