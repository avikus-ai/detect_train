#!/bin/bash

python train.py \
--project HiNAS-IR \
--name v5l_1280_avikus \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir-avikus.yaml \
--cfg models/yolov5l-ir.yaml \
--weights v5l_1280_FLIR_best.pt \
--device 3 \
--imgsz 1280 \
--epochs 100 \
--batch-size 4