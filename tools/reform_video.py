"""
----------------------------------------------------------------------------
Filename: reform_video.py
Path: tools/reform_video.py
Date: 10.10.2023
Author: youngjae.you
Purpose: 코덱에 문제가 있는 영상에 대해서 한 프레임씩 파싱해서 비디오를 재생성하는 스크립트
History:
   
----------------------------------------------------------------------------
"""

import os
import cv2

VIDEO_PATH = "assets/apple.mp4"

OUTPUT_PATH = os.path.splitext(VIDEO_PATH)[0] + '_reformed.mp4'

def reform_video(input_path, output_path):
    # 비디오 캡쳐 객체 생성
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {input_path}")
        return

    # 비디오의 프레임 크기 및 FPS 얻기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()

        # 프레임이 더이상 없으면 종료
        if not ret:
            break

        out.write(frame)

    cap.release()
    out.release()
    print(f"Reformed video saved to {output_path}")

if __name__ == "__main__":
    reform_video(VIDEO_PATH, OUTPUT_PATH)





