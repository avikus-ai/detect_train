#!/bin/bash

python3 detect.py \
    --data data/report-class.yaml \
    --source data/report-class.yaml \
    --weights best_park.pt \
    --device 0 \
    --imgsz 544 960 \
    --save-txt \
    --save-conf

