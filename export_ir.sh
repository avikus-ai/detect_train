#!/bin/bash
python export.py \
	--data \
    /data01/HiNAS-DATA/CV-MODEL/2023-IR-YOLOv5l/data-ir-avikus.yaml \
	--weights \
	/data01/HiNAS-DATA/CV-MODEL/2023-IR-YOLOv5l/b1_704_1280_ir_best.pt \
	--include \
	onnx \
	--imgsz \
	704 1280 \
	--opset \
	16 \
	--batch-size \
	1 \
	--simplify \
    --inplace
