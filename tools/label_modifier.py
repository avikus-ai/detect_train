"""
----------------------------------------------------------------------------
Filename: label_modifier.py
Path: tools/label_modifier.py
Date: 10.19.2023
Author: junghyun.hwang youngjae.you
Purpose: 기존 라벨 파일에 대해서 라벨을 수정하는 스크립트 
History:
   - 
----------------------------------------------------------------------------
"""

import os
import numpy as np

from pathlib import Path
from glob import glob
from tqdm import tqdm

labels_root_path = '/data/NeuBoat/Avikus/NEW_KIMPO/labels'

"""
2023.10.19 라벨 현황
0: person
1: boat
2: jetski
3: canoe
4: buoy
5: channel marker
"""

# 기존 라벨: 새로운 라벨
cls_map = {
    0:0,
    1:1,
    2:1,
    3:1,
    4:2,
    5:3,
}

names = \
    {0: 'Person', 
     1: 'Boat', 
     2: 'Buoy', 
     3: 'Channel Marker'}


def modify_label(label_file: str,
                 label_dir: str) -> None:
    """
    단독 label_file 수정
    label_file: 수정할 라벨 파일 경로
    label_dir: 수정된 라벨 파일을 저장할 폴더 경로
    """
    
    label_name = os.path.basename(label_file)
    labels = np.loadtxt(label_file)
    
    if labels.size <= 0:
        return
    
    if labels.ndim == 1:
        labels = np.expand_dims(labels, 0)
    idxs = (labels[:, [0]] == np.array(list(cls_map.keys()))).any(1)
    labels = labels[idxs]
    labels[:, 0] = np.array([cls_map[int(c)] for c in labels[:, 0]], dtype=float)
    
    with open(os.path.join(label_dir, label_name), 'w') as f:
        np.savetxt(f, labels, fmt='%.4f')


def modify_labels(label_dir: str,
                  is_splitted: bool,
                  in_place: bool = False) -> None:
    """
    사용하는 경우: 특정 폴더의 하위 폴더들을 한 번에 수정하려고 할 때
    is_splitted: train, val로 나누어져 있는지 여부
    in_place: 기존 라벨 파일을 수정할지, 새로운 폴더에 저장할지 여부
    example:
        label_dir = '/workspace/data/Avikus/TAMPA'
    """
    
    tasks = ['train', 'val']
    
    if is_splitted:
        for t in tasks:
            label_files = list(glob(os.path.join(label_dir, t, '*.*')))
            
            if not in_place:
                l_dir = Path(label_dir).parent / 'labels_modified' / t
                l_dir.mkdir(parents=True)
            else:
                l_dir = Path(label_dir) / t
            
            for label_file in tqdm(label_files):
                modify_label(label_file, l_dir)
                
    else:
        label_files = list(glob(os.path.join(label_dir, '*.*')))
        
        if not in_place:
            l_dir = Path(label_dir).parent / 'labels_modified'
            l_dir.mkdir(parents=True)
        else:
            l_dir = Path(label_dir)
        
        for label_file in tqdm(label_files):
            modify_label(label_file, l_dir)


def modify(top_dir: str) -> None:
    """
    사용하는 경우: Avikus, Open 이하의 폴더들을 한 번에 수정하려고 할 때
    example:
        top_dir = '/workspace/data/Avikus'
    """

    valid_dirs = [d for d in os.listdir(top_dir) if 'labels' in os.listdir(os.path.join(top_dir, d))]

    for valid_dir in valid_dirs:
        label_dir = os.path.join(top_dir, valid_dir, 'labels')
        
        is_splitted = 'train' in os.listdir(label_dir)
        
        modify_labels(label_dir, is_splitted)
        
        
if __name__ == '__main__':
    # top_dir = '/workspace/data/Avikus'
    # modify(top_dir)

    # label_dir = '/data/NeuBoat/Avikus/NEW_KIMPO/labels'
    # modify_labels(label_dir, is_splitted=True, in_place=False)