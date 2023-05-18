#!/bin/bash

python -m torch.distributed.run --nproc_per_node 2 train.py \
--project HiNAS-IR \
--name v5l_1280_avikus_add_train \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir-avikus.yaml \
--cfg models/yolov5l-ir.yaml \
--weights HiNAS-IR/v5l_1280_FLIR/weights/last.pt \
--device 0,1 \
--imgsz 1280 \
--epochs 300 \
--batch-size 4 
