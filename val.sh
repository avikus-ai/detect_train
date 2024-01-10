#!/bin/bash

python3 val.py \
    --data data/2-class-revised+s3.yaml \
    --weights  ./best_park.pt \
    --device 1 \
    --imgsz 960 \
    --conf-thres 0.4 \
    --iou-thres 0.45 \
    # --save-json