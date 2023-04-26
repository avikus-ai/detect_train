#!/bin/bash

python train.py \
--project HiNAS-IR \
--name v5sface_1280_avikus \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir-avikus.yaml \
--cfg models/yolov5s_face_ir.yaml \
--weights HiNAS-IR/v5face_1280_FLIR/weights/best.pt \
--device 3 \
--imgsz 1280 \
--epochs 100 \
--batch-size 16