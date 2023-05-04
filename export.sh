#!/bin/bash

python export.py \
    --data \
    data/eo-slicing.yaml \
    --imgsz \
    704 1280 \
    --weights \
    /data01/HiNAS-DATA/CV-MODEL/2023-EO-YOLOv5Face/best.pt \
    --batch-size \
    1 \
    --device \
    0 \
    --include \
    onnx \
    --opset \
    16 \
    --simplify \
    --inplace \
