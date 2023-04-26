#!/bin/bash
python export.py \
	--data \
    data/data-ir-avikus.yaml \
	--weights \
	HiNAS-IR/v5l_1280_avikus/weights/best.pt \
	--include \
	onnx \
	--imgsz \
	704 1280 \
	--opset \
	16 \
	--batch-size \
	1
