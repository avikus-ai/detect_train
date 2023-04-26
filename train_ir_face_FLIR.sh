#!/bin/bash

python train.py \
--project HiNAS-IR \
--name v5face_1280_FLIR \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir-FLIR.yaml \
--cfg models/yolov5s_face_ir.yaml \
--weights checkpoint/v5sface-cocopretrain/weights/best.pt \
--device 3 \
--imgsz 1280 \
--epochs 100 \
--batch-size 8