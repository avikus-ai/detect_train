#!/bin/bash

python classify/predict.py \
    --source \
    sources/image/crops/Vessel \
    --weights \
    checkpoint/eo_classification2.pt \
    --device \
    0
