"""
----------------------------------------------------------------------------
Filename: general_json2yolo.py
Path: general_json2yolo.py
Date: 2024.02.15
Author: youngjae.you
Purpose: COCO json to YOLO format (Instance Segmentation)
History:
   - 
Usage:
    - 현재는 Main 함수에서 `REGION = 'NEW_BUSAN'`을 직접 변경해서 사용 가능
    - Tampa
        - python general_json2yolo.py --ann-dir /data/NeuBoat/Batch1_3223_img/TAMPA/labels --save-dir /data/NeuBoat/Batch1_3223_img/TAMPA
        
    - Wangsan
        - python general_json2yolo.py --ann-dir /data/NeuBoat/Batch1_3223_img/NEW_WANGSAN/labels --save-dir /data/NeuBoat/Batch1_3223_img/NEW_WANGSAN
----------------------------------------------------------------------------
"""

import json
import argparse

from collections import defaultdict
from utils import *

def convert_coco_json(json_dir="../coco/annotations/", save_dir_path = "", use_segments=False, cls91to80=False):
    save_dir = make_dirs(save_dir_path)  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels"
        fn.mkdir(exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                
                # bbox가 비어있지 않다면
                if len(ann["bbox"]) > 0:
                    box = np.array(ann["bbox"], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    
                    cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                        
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    # 거리가 최소인 
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            # idx_list: [[5], [12, 0], [7]]
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                # 첫번째와 마지막 세그먼트를 제외한 나머지 세그먼트들은 두개의 인덱스를 가지고 있음
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    # segments[i] : (N, 2)
                    segments[i] = segments[i][::-1, :]

                # np.roll -idx[0] 행만큼 이동
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                # 앞 점과 뒷 점을 같게 만들기 위해
                segments[i] = np.concatenate([segments[i], segments[i][:1]])

                # deal with the first segment and the last one
                # 앞 뒤
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            # Reverse로 
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s




if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Convert COCO json to YOLO format")
    parser.add_argument("--source", type=str, default="COCO", help="source dataset")
    parser.add_argument("--ann-dir", type=str, default="annotations", help="annotation directory")
    parser.add_argument("--save-dir", type=str, default="labels", help="output directory")
    
    args = parser.parse_args()
    
    if args.source == "COCO":
        convert_coco_json(
            # "../datasets/coco/annotations",
            args.ann_dir,
            save_dir_path = args.save_dir,
            use_segments=True,
            cls91to80=False,
        )