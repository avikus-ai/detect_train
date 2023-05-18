#!/bin/bash

python detect.py \
    --data \
    data/data-ir-s3.yaml \
    --source \
    data/data-ir-s3.yaml \
    --weights \
    /data01/HiNAS-DATA/CV-MODEL/Maersk/IR/maersk_ir.pt \
    --device \
    0 \
    --imgsz \
    1280 \
    --conf-thres \
    0.4 \
    --project \
    2023-IR-S3-Detect \
    --save-txt
