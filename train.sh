python -m torch.distributed.run --nproc_per_node 2 train.py \
--data aiboat.yaml \
--project AiBOAT_dev_coco_val \
--cfg yolov5s.yaml \
--img-size 960 \
--epochs 3 \
--batch-size 6 \
--device 0,1