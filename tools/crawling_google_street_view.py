"""
----------------------------------------------------------------------------
Filename: crawling_google_street_view.py
Path: tools/crawling_google_street_view.py
Date: 10.19.2023
Author: junghyun.hwang youngjae.you
Purpose: 구글 스트리트 뷰 이미지 크롤링
History:
   - 
----------------------------------------------------------------------------
"""

import numpy as np
import urllib
import os

region = 'cannes'
save_dir = f'images/{region}'
# cvteam key
key = ''

LATITUDE_MIN = 43.5506065
LATITUDE_MAX = 43.5507673
LATITUDE_NUM = 10

LONGITUDE_MIN = 7.0124943
LONGITUDE_MAX = 7.0177375
LONGITUDE_NUM = 10

HEADING_MIN, HEADING_MAX = 0, 360
HEADING_NUM = 5

PITCH_MIN, PITCH_MAX = -10, 10
FOV_MIN, FOV_MAX = 80, 120

# 있으면 기존 것 삭제
if os.path.exists(save_dir):
    os.system(f'rm -rf {save_dir}')
os.makedirs(save_dir, exist_ok=True)

lats = np.linspace(LATITUDE_MIN, LATITUDE_MAX, LATITUDE_NUM)
longs = np.linspace(LONGITUDE_MIN, LONGITUDE_MAX, LONGITUDE_NUM)
headings = np.linspace(HEADING_MIN, HEADING_MAX, HEADING_NUM)

save_count = 0

for lat in lats:
    for long in longs:
        for heading in headings:
            pitch = np.random.randint(PITCH_MIN, PITCH_MAX)
            fov = np.random.randint(FOV_MIN, FOV_MAX)
            url = f'https://maps.googleapis.com/maps/api/streetview?size=1080x720&location={lat},{long}&heading={heading}&pitch={pitch}&fov={fov}&key={key}'
            f = f'{region}_{lat}_{long}_{pitch}_{heading}_{fov}.jpg'

            print('Save image: ', f, '(', save_count, ')')
            
            urllib.request.urlretrieve(url, os.path.join(save_dir, f))  # 받아온 이미지 저장
            
            save_count += 1