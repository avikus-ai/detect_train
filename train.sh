#!/bin/bash

python3 train.py \
    --hyp data/hyps/eo.yaml \
    --data 'data/3-class-revised+s3.yaml' \
    --cfg models/yolov5m.yaml \
    --weights yolov5m.pt \
    --device '0,1'\
    --imgsz 960 \
    --epochs 150 \
    --batch-size 44 \
    --project '3-class-maersk' \
    --name 'v5m_detect-train_avi'\
    --workers 16