#!/bin/bash

python train.py \
--project HiNAS-IR \
--name v5l_1280_avikus_noPretrained \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir-avikus.yaml \
--cfg models/yolov5l-ir.yaml \
--weights yolov5l.pt \
--device 2 \
--imgsz 1280 \
--epochs 100 \
--batch-size 4