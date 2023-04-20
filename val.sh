#!/bin/bash

python val.py \
    --data \
    data/eo-slicing.yaml \
    --weights \
    slicing_best.pt \
    --device \
    1 \
    --imgsz \
    1280 \
    --conf-thres \
    0.25 \
    --iou-thres \
    0.45
