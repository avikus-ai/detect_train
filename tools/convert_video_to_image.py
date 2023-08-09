"""
----------------------------------------------------------------------------
Filename: convert_video_to_image.py
Path: tools/convert_video_to_image.py
Date: 08.09.2023
Author: junghyun.hwang youngjae.you
Purpose: 비디오 프레임 추출 (sec를 지정해서 INTERVAL을 설정할 수 있음)
History:
----------------------------------------------------------------------------
"""

import os
import sys
import argparse
import re
from multiprocessing import Pool
from glob import glob
from pathlib import Path
import numpy as np
import cv2

def extract(vid_path: str,
            out_dir: str,
            sec: int = 1):
    vid_path, out_f_name = vid_path
    out_dir = Path(out_dir) / out_f_name
    out_dir.mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    f_num = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        frame_name = out_dir / f'{out_f_name}_{f_num:05}.jpg'
        cv2.imwrite(str(frame_name), image)
        f_num += 1
        count += sec
        success, image = vidcap.read()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vids-dir', type=str, default=os.path.join(os.getenv('HOME'), 'Videos/4K Video Downloader'))
    parser.add_argument('--out-dir', type=str, default=os.path.join(os.getenv('HOME'), 'Videos/frames'))
    
    args = parser.parse_args()
    vids = glob(os.path.join(args.vids_dir, '*.mp4'))
    
    existing_vids = os.listdir(args.out_dir)
    out_dir = args.out_dir
    vids_to_extract = []

    for i, vid_out_name in enumerate(vids):
        
        vid_out_name = Path(vid_out_name).stem
        vid_out_name = re.sub(r'[\s]', '_', vid_out_name)
        vid_out_name = re.sub(r'[^\w]', '', vid_out_name)

        # 비디오 이름으로 폴더가 생성됨
        if vid_out_name not in existing_vids:
            (Path(out_dir) / vid_out_name).mkdir(parents=True, exist_ok=True)
            vids_to_extract.append((vids[i], vid_out_name))

    num_vids = len(vids_to_extract)
    
    print(f"number of videos: {num_vids}")
    with Pool(os.cpu_count()) as pool:
        pool.starmap(extract, zip(vids_to_extract, [out_dir] * num_vids, [1] * num_vids))