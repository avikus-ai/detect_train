#!/bin/bash
python export.py \
	--weights HiNAS-IR/v5l_1280_FLIR/weights/best.pt \
	--include onnx \
	--simplify \
	--imgsz 1280 1280 \
	--data data/data-ir.yaml \
	--opset 16
