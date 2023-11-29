#!/bin/bash

python detect.py \
    --source \
    sources/video/imgs1.mp4 \
    --weights \
    checkpoint/eo_2class_July.pt \
    --apply-cls \
    --cls-weights \
    checkpoint/eo_classification2.pt \
    --device \
    0 \
    --imgsz \
    960 \
    --conf-thres \
    0.3 \
    --project \
    2stage-test \
    --view-img
