python train.py \
--data aiboat.yaml \
--project AiBOAT_dev \
--name random_crop \
--weights yolov5n.pt \
--img-size 960 \
--epochs 2 \
--batch-size 3 \
--device 0 \
--stop-coco-eval
# python -m torch.distributed.run --nproc_per_node 2 train.py \
# --cfg hub/yolov5m-transformer.yaml \