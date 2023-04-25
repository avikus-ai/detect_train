#!/bin/bash

python val.py \
    --data \
    data/data-ir-avikus.yaml \
    --weights \
    HiNAS-IR/v5l_1280_avikus/weights/best.pt \
    --device \
    3 \
    --imgsz \
    1280 \
    --conf-thres \
    0.001 \
    --iou-thres \
    0.6