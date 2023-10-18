"""
----------------------------------------------------------------------------
Filename: create_video.py
Path: tools/create_video.py
Date: 10.18.2023
Author: youngjae.you
Purpose: 이미지 폴더를 비디오 파일로 만드는 작업
History:
   - re를 이용한 정규식으로 파일명을 정렬하고, 파일명을 숫자로 변경하여 정렬
----------------------------------------------------------------------------
"""

import os
import cv2
from tqdm import tqdm
import re

FPS = 10
IMAGE_DIR = '/media/aicads/T7/Avikus/Commercial/data/diffusion/pix2pix/Maersk_data/2023-01-15_11/cloud1'
OUTPUT_VIDEO_NAME = 'maersk_230115_11_cloud1.mp4'
# 처리할 이미지 장수 제한
MAX_IMAGES = 1220
# MAX_IMAGES = -1
# 정렬 방식 (이미지 파일명이 00001, 00002가 아닌 1, 2, 3, ... 10, 11, ... 100, 101, ... 순으로 정렬되어 있을 때)
USE_RE = True

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    리스트를 자연스러운 순서로 정렬하기 위한 헬퍼 함수.
    숫자를 찾아서 해당 숫자를 정수로 변환(atoi)하여 리스트를 생성
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def create_video_from_images(images_dir, fps=30):
    # Loop over all subdirectories in the images directory
    for root, _, files in os.walk(images_dir):
        # Check if the directory contains JPG or PNG files
        image_files = [os.path.join(root, file) for file in files if file.endswith(('.jpg', '.png'))]
        if not image_files:
            continue

        # Set the output video file path
        output_video_path = os.path.join(root, OUTPUT_VIDEO_NAME)

        # Open the first image to get the frame size
        first_image = cv2.imread(image_files[0])
        frame_size = (first_image.shape[1], first_image.shape[0])

        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        if USE_RE:
            image_files.sort(key=natural_keys)
        else:
            image_files = sorted(image_files)
        
        # Loop over the image files and write each frame to the video
        for image_file in tqdm(image_files[:MAX_IMAGES]):
            image = cv2.imread(image_file)
            video_writer.write(image)

        # Release the video writer and print a message
        video_writer.release()
        print(f"Video saved to {output_video_path}")

# Example usage
create_video_from_images(IMAGE_DIR, fps=10)