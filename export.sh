#!/bin/bash

python export.py \
    --data \
    data/eo-slicing-batch02.yaml \
    --imgsz \
    544 960 \
    --weights \
    Seaspan/EO/seaspan_eo.pt \
    --batch-size \
    4 \
    --device \
    1 \
    --include \
    onnx \
    --opset \
    12 \
    --simplify \
    --inplace \
