#!/bin/bash

python train.py \
    --hyp \
    data/hyps/hyp.scratch-low.yaml \
    --data \
    data/NeuBoat_runpod.yaml \
    --weights \
    yolov5l.pt \
    --device \
    0 \
    --imgsz \
    960 \
    --epochs \
    100 \
    --batch-size \
    32 \
    --project \
    NeuBoat \
    --name \
    yolov5l

