#!/bin/bash

python detect.py \
    --data \
    data/data-ir-avikus.yaml \
    --source \
    data/data-ir-avikus.yaml \
    --weights \
    HiNAS-IR/v5l_1280_avikus/weights/best.pt \
    --device \
    3 \
    --imgsz \
    1280 \
    --conf-thres \
    0.25 \
    --project \
    2023-IR-Detect