python -m torch.distributed.run --nproc_per_node 3 train.py \
--data aiboat.yaml \
--project AiBOAT_dev \
--name sahi \
--weights yolov5s.pt \
--img-size 960 \
--epochs 50 \
--batch-size 120 \
--device 1,2,3
# python -m torch.distributed.run --nproc_per_node 2 train.py \
# --cfg hub/yolov5m-transformer.yaml \