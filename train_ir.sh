#!/bin/bash

python -m torch.distributed.run --nproc_per_node 2 \
train.py \
--project HiNAS-IR \
--name v5l_1280_FLIR \
--img_type ir \
--hyp data/hyps/hyp-ir.yaml \
--data data/data-ir.yaml \
--cfg models/yolov5l-ir.yaml \
--weights yolov5l.pt \
--device 0,1 \
--imgsz 1280 \
--epochs 100 \
--batch-size 8
