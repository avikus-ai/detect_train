#!/bin/bash

python detect.py \
    --source \
    sources/hannara.mp4 \
    --weights \
    checkpoint/eo_2class.pt \
    --apply-cls \
    --cls-weights \
    checkpoint/eo_classification.pt \ 
    --device \
    0 \
    --imgsz \
    960 \
    --conf-thres \
    0.3 \
    --project \
    2stage-test
