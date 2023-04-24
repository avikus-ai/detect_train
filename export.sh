#!/bin/bash

python export.py \
    --data \
    data/eo-slicing.yaml \
    --imgsz \
    960 1280 \
    --weights \
    Seaspan_project/v5sface_1280_wv5sface_mosaic1.0_slicing/weights/best.pt \
    --batch-size \
    1 \
    --device \
    0 \
    --include \
    onnx \
    --opset \
    16
