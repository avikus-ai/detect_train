#!/bin/bash

python val.py \
    --data \
    data/eo-slicing-batch02.yaml \
    --weights \
    Seaspan/EO/seaspan_eo.onnx \
    --device \
    1 \
    --imgsz \
    960 \
    --conf-thres \
    0.001 \
    --iou-thres \
    0.6
