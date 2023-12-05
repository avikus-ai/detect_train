"""
----------------------------------------------------------------------------
Filename: bbox_drawer.py
Path: tools/bbox_drawer.py
Date: 10.19.2023
Author: junghyun.hwang youngjae.you
Purpose: 이미지 파일과 라벨 파일을 읽어서 마스크 이미지 저장 
History:
   - NeuBoat.yaml 기반에서 Draw
     - data/NeuBoat.yaml
Usage:
    - 현재는 라벨을 그리고자 하는 이미지 폴더를 하나씩 직접 넣어줘야 함 
----------------------------------------------------------------------------
"""

import os
import numpy as np
from tqdm import tqdm
from random import randint

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 랜덤한 색상을 생성하는 함수
def random_color():
    # 최대한 밝은 색상 위주로 생성하도록 한다
    return (randint(0, 150), randint(0, 150), randint(0, 150))

# 카테고리별로 색상을 미리 할당합니다.
category_colors = {category: random_color() for category in names}

def bbox_drawer(img_dir, label_dir, out_dir, names):
    img_files = os.listdir(img_dir)
    no_label_imgs = []
    no_label_img_cnt = 0
    for img_file in tqdm(img_files):
        label_file = f'{img_file.rsplit(".", 1)[0]}.txt'
        
        ext = img_file.rsplit(".", 1)[-1]
        if ext.lower() not in ('jpg', 'jpeg', 'png'):
            continue
        
        img_file_path = img_dir / img_file
        label_file_path = label_dir / label_file
        pil_img = Image.open(img_file_path)
        
        # if image is portrait, rotate it
        if pil_img.size[0] < pil_img.size[1]:
            pil_img = pil_img.rotate(90, expand=True)
        
        pil_draw = ImageDraw.Draw(pil_img)
        width, height = pil_img.size

        try:
            labels = np.loadtxt(label_file_path)
        except Exception as e:
            print(f"Error loading label file: {e}")
            no_label_img_cnt += 1
            no_label_imgs.append(img_file)
            labels = np.array([])

        if labels.size > 0:
            if labels.ndim == 1:
                labels = np.expand_dims(labels, 0)
            x1 = (labels[:, 1] - labels[:, 3] / 2) * width
            y1 = (labels[:, 2] - labels[:, 4] / 2) * height
            x2 = (labels[:, 1] + labels[:, 3] / 2) * width
            y2 = (labels[:, 2] + labels[:, 4] / 2) * height

            for label in np.stack((labels[:, 0], x1, y1, x2, y2), axis=1).tolist():
                cat, xx1, yy1, xx2, yy2 = label
                category_id = int(cat)
                # 미리 할당된 색상 사용
                box_color = category_colors.get(category_id, (255, 255, 255))  # 카테고리가 없는 경우 흰색 사용
                pil_draw.rectangle((xx1, yy1, xx2, yy2), outline=box_color, width=2)
                text = f'{names[category_id]}'
                # 텍스트 배경을 생성합니다.
                text_size = ImageFont.truetype('arial.ttf', size=20).getsize(text)
                pil_draw.rectangle((xx1, yy1 - 22, xx1 + text_size[0], yy1), fill=box_color)
                # 텍스트를 그립니다. (흰색으로 보이도록 합니다.)
                pil_draw.text((xx1, yy1 - 22), text, fill=(255,255,255), font=ImageFont.truetype('arial.ttf', size=20))
        
        img_out_file_path = out_dir / img_file
        pil_img.save(img_out_file_path)

    print(f'{no_label_img_cnt} image file dont have labels')
    return no_label_imgs, no_label_img_cnt

if __name__ == '__main__':
    
    data_root_path = '/data/NeuBoat'
    label_folder_name = 'labels.bak'
    
    data = \
    {
        'Avikus': {
            'NEW_BUSAN': ['train', 'val'],
            'NEW_KIMPO': [],
            'NEW_WANGSAN': ['train', 'val'],
            'Wangsan': ['train', 'val'],
            
            'TAMPA': ['train', 'val'],
            'FLL': ['train', 'val'],
            'ibex': [],
        },
        'Open': {
            
        }
    }
    
    # names = {0: 'Person', 1: 'Boat', 2: 'Buoy', 3: 'Channel Marker'}
    names = {0: 'Person', 1: 'Boat', 2: 'Jetski', 3: 'Canoe', 4: 'Buoy', 5: 'Marker'}
    
    for k1, v1 in data.items():
        for k2, v2 in v1.items():
            if v2 == []:
                IMG_DIR = Path(f'{data_root_path}/{k1}/{k2}/images/')
                LABEL_DIR = Path(f'{data_root_path}/{k1}/{k2}/{label_folder_name}/')
                OUT_DIR = Path(f'{data_root_path}/{k1}/{k2}/masked_images')
            else:
                for v in v2:
                    IMG_DIR = Path(f'{data_root_path}/{k1}/{k2}/images/{v}')
                    LABEL_DIR = Path(f'{data_root_path}/{k1}/{k2}/{label_folder_name}/{v}')
                    OUT_DIR = Path(f'{data_root_path}/{k1}/{k2}/masked_images_{v}')
                
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bbox_drawer(IMG_DIR, LABEL_DIR, OUT_DIR, names)