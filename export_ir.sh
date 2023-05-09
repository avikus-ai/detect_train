#!/bin/bash
python export.py \
	--data \
    ./data/data-ir-avikus.yaml \
	--weights \
	./Seaspan/IR/seaspan_ir.pt \
	--include \
	onnx \
	--imgsz \
	704 1280 \
	--opset \
	12 \
	--batch-size \
	1 \
	--simplify \
    --inplace
