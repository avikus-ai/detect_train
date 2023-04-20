#!/bin/bash

python train.py \
    --hyp \
    data/hyps/eo.yaml \
    --data \
    data/eo-slicing.yaml \
    --cfg \
    models/yolov5s6.yaml \
    --weights \
    yolov5s6.pt \
    --device \
    1 \
    --imgsz \
    1280 \
    --epochs \
    50 \
    --batch-size \
    32 \
    --project \
    Seaspan_project \
    --name \
    v5s6_1280_wv5s6_mosaic1.0_slicing

