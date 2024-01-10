#!/bin/bash

<<<<<<< HEAD
python3 detect.py \
    --data data/report-class.yaml \
    --source data/report-class.yaml \
    --weights best_park.pt \
    --device 0 \
    --imgsz 544 960 \
    --save-txt \
    --save-conf

=======
python detect.py \
    --source \
    sources/hannara.mp4 \
    --weights \
    checkpoint/eo_2class.pt \
    --apply-cls \
    --cls-weights \
    checkpoint/eo_classification.pt \ 
    --device \
    0 \
    --imgsz \
    960 \
    --conf-thres \
    0.3 \
    --project \
    2stage-test
>>>>>>> b043955318df861dba0f03b2d806bfc868633a2e
