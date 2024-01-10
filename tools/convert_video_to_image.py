"""
----------------------------------------------------------------------------
Filename: convert_video_to_image.py
Path: tools/convert_video_to_image.py
Date: 08.09.2023
Author: junghyun.hwang youngjae.you
Purpose: 비디오 프레임 추출 (sec를 지정해서 INTERVAL을 설정할 수 있음)
History:
   - 23.08.21 예외 처리 추가 및 버그 수정 by youngjae.you
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
    
    # 예외 처리 (비디오 파일이 없는 경우)
    if not vidcap.isOpened():
        print(f"Failed to open video: {vid_path}")
        return
    
    video_length = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_FPS)  # in seconds

    success, image = vidcap.read()
    count = 0
    f_num = 0
    
    while success:
        if image is None or image.size == 0:
            print(f"Invalid frame detected at {count} seconds in video: {vid_path}")
            success = False  # Invalid frame, but continue the loop.
            continue
        else:
            # Check if the next position is beyond the video's length
            if count * 1000 > video_length * 1000:
                print(f"Reached end of video at {count} seconds.")
                break
            
            # 현재 프레임의 위치를 msec 단위로 설정
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
            frame_name = out_dir / f'{out_f_name}_{f_num:05}.jpg'
            cv2.imwrite(str(frame_name), image)
            f_num += 1
            count += sec
        
        try:
            success, image = vidcap.read()
        except Exception as e:
            print(f"Error reading frame at {count} seconds in video: {vid_path}. Error: {e}")
            success = False  # Stop the loop.
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--vids-dir', type=str, default=os.path.join(os.getenv('HOME'), 'Videos/4K Video Downloader'))
    parser.add_argument('--out-dir', type=str, default=os.path.join(os.getenv('HOME'), 'Videos/frames'))
    parser.add_argument('--sec', type=int, default=1)
    
    args = parser.parse_args()
    vids = glob(os.path.join(args.vids_dir, '*.mp4'))
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    
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
        pool.starmap(extract, zip(vids_to_extract, [out_dir] * num_vids, [args.sec] * num_vids))
