#!/bin/bash

python detect.py \
    --data \
    data/eo-slicing.yaml \
    --source \
    data/eo-slicing.yaml \
    --weights \
    slicing_best.pt \
    --device \
    1 \
    --imgsz \
    1280 \
    --conf-thres \
    0.5 \
    --project \
    2023-EO-Detect
