python -m torch.distributed.run --nproc_per_node 2 train.py \
--data aiboat.yaml \
--project AiBOAT_dev \
--cfg yolo5gm_face.yaml \
--img-size 640 \
--epochs 3 \
--batch-size 6 \
--device 0,1