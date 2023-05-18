############## make slicing image - discard under 20 pix version (working) ############################
import torch
import torchvision
from torchvision import transforms
import PIL
from torchvision.datasets.folder import pil_loader
import torch.nn as nn
import random
import numpy as np
import glob
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
import io
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import ExifTags, Image, ImageOps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    if im.shape[0] == 3 or im.shape[0] == 1:
        im = im.transpose((1, 2, 0))
        im2 = im.copy()
        im2[:,:,0] = im[:,:,2]
        im2[:,:,2] = im[:,:,0]
        im = im2
    height, width, _ = im.shape
    im2 = im.copy()
    im2[:,:,0] = im[:,:,2]
    im2[:,:,2] = im[:,:,0]

    im = im2
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height
    idx = 0
    # Create a Rectangle potch
    for box in boxes:
        # class_idx = classes[int(box[0])]
        box = box[1:]
       
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add the label text
        # plt.text(
        #     upper_left_x * width, upper_left_y * height,
        #     idx,
        #     fontdict=dict(color="white"),
        #     bbox=dict(facecolor="red", edgecolor="red", pad=0)
        # )
        idx = idx + 1

    # ax.set_frame_on(False)
    ax.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches="tight",format='png', pad_inches = 0)

    im = Image.open(img_buf)
    im.show()
    return im

def get_xy_cord(img_w, img_h, box):
    box = box[1:]

    min_x = (box[0] - box[2] / 2) * img_w
    min_y = (box[1] - box[3] / 2) * img_h
    width = box[2] * img_w
    height = box[3] * img_h
    max_x = min_x + width
    max_y = min_y + height
    if max_x > img_w:
        max_x = img_w-1
    if max_y > img_h:
        max_y = img_h-1
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0
    return min_x, min_y, max_x, max_y

def save_array_to_file(arr, filename):
    """
    This function takes an array and a filename as input and saves the array to a text file with the given filename.
    """
    # arr = list(arr)
    with open(filename, 'w') as f:
        # Open the file with the given filename in write mode
        for item in arr:
            for value in item:
                f.write(str('{:.4f}'.format(value)) + ' ')
            
            # Loop through each item in the array
            f.write('\n')
            # Write the item to the file followed by a newline character
        print(f'The array has been saved to {filename}')
        # Print a message indicating that the array has been saved to the file

min_w = 20
max_area = 0.05
# file_list = ['S3']
file_list = ['2939','ccnuri','GORAE/221020','GORAE/221026','GORAE/221028', 'hannara/busan','hannara/incheon_arriving', 'hannara/incheon_berth','KHOPE', 'YMwinner']
for file in file_list:
    # img_dataset_folder = sorted(glob.glob("/data/" + file + "/images/train/*.jpg"))
    # txt_dataset_folder = sorted(glob.glob("/data/" + file + "/labels/train/*.txt"))
    img_dataset_folder = sorted(glob.glob("/data/batch01_new/SINGLE_CLASS/" + file + "/images/train/*.jpg"))
    txt_dataset_folder = sorted(glob.glob("/data/batch01_new/SINGLE_CLASS/" + file + "/labels/train/*.txt"))

    for d_idx in range(len(img_dataset_folder)):
        with open(txt_dataset_folder[d_idx]) as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        img_file_name = img_dataset_folder[d_idx].split("train/")[-1].split('.jpg')[0]
        txt_file_name = txt_dataset_folder[d_idx].split("train/")[-1].split('.txt')[0]
        sample = torchvision.transforms.ToTensor()(pil_loader(img_dataset_folder[d_idx]))
        c,h,w = sample.shape
    
        s1 = sample[:,:,:int(w/3)]
        s2 = sample[:,:,int(w/3):2*int(w/3)]
        s3 = sample[:,:,2*int(w/3):w]
        l1 = []
        l2 = []
        l3 = []
    
        for p in range(len(lb)):
            min_x, min_y, max_x, max_y = get_xy_cord(w, h, lb[p])

            if max_x >= 0 and max_x < int(w/3) and min_x >= 0 and min_x < int(w/3): # 모두 다 포함
                l1.append([lb[p][0], ((max_x+min_x)/2)/int(w/3), ((max_y+min_y)/2)/h, (max_x-min_x)/int(w/3), (max_y-min_y)/h])

            elif min_x >= 0 and min_x < int(w/3) and max_x >= int(w/3) and max_x < int(w/3)*2: # 1st, 2nd frame에 걸침
                b1_width = (int(w/3)-1)-min_x
                b1_height = (max_y-min_y)
                cx1 = (((int(w/3)-1)+min_x)/2)/int(w/3)
                cy1 = ((max_y+min_y)/2)/h
                bw1 = ((int(w/3)-1)-min_x)/int(w/3)
                bh1 = (max_y-min_y)/h

                if bw1>(min_w/int(w/3)) and bh1>0:
                    if bw1 * bh1 > (max_area):
                        l1.append([lb[p][0], cx1, cy1, bw1, bh1])
            
                b2_width = max_x - int(w/3)
                b2_height = (max_y-min_y)
                cx21 = ((b2_width)/2)/int(w/3)
                cy21 = ((max_y+min_y)/2)/h
                bw21 = (b2_width)/int(w/3)
                bh21 = (max_y-min_y)/h

                if bw21>(min_w/int(w/3)) and bh21>0:
                    if bw21 * bh21 > (max_area):
                        l2.append([lb[p][0], cx21, cy21, bw21, bh21])

            elif max_x >= int(w/3) and max_x < int(w/3)*2 and min_x >= int(w/3) and min_x < int(w/3)*2:
                max_x_temp = max_x - int(w/3)
                min_x_temp = min_x - int(w/3)
                l2.append([lb[p][0], ((max_x_temp+min_x_temp)/2)/int(w/3), ((max_y+min_y)/2)/h, (max_x_temp-min_x_temp)/int(w/3), (max_y-min_y)/h])

            elif min_x >= int(w/3) and min_x < int(w/3)*2 and max_x >= int(w/3)*2 and max_x < w:# 2nd, 3rd frame에 걸침
                b2_width = ((2*int(w/3))-1)-min_x
                b2_height = (max_y-min_y)
                cx22 = ((((2*int(w/3))-1)+(min_x-(2*int(w/3))))/2)/int(w/3)
                cy22 = ((max_y+min_y)/2)/h
                bw22 = b2_width/int(w/3)
                bh22 = (max_y-min_y)/h

                if bw22>(min_w/int(w/3)) and bh22>0:
                    if bw22 * bh22 > (max_area):
                        l2.append([lb[p][0], cx22, cy22, bw22, bh22])
            
                b3_width = max_x - 2*int(w/3)
                b3_height = (max_y-min_y)
                cx3 = ((b3_width)/2)/int(w/3)
                cy3 = ((max_y+min_y)/2)/h
                bw3 = (b3_width)/int(w/3)
                bh3 = (max_y-min_y)/h

                if bw3>(min_w/int(w/3)) and bh3>0:
                    if bw3 * bh3 > (max_area):
                        l3.append([lb[p][0], cx3, cy3, bw3, bh3])
        
            elif max_x >= int(w/3)*2 and max_x < w and min_x >= int(w/3)*2 and min_x < w:
                max_x_temp = max_x - (2*int(w/3))
                min_x_temp = min_x - (2*int(w/3))
                l3.append([lb[p][0], ((max_x_temp+min_x_temp)/2)/(int(w/3)), ((max_y+min_y)/2)/h, (max_x_temp-min_x_temp)/(int(w/3)), (max_y-min_y)/h])  
        
        l1 = np.array(l1)
        l2 = np.array(l2)
        l3 = np.array(l3)

        save_folder = '2023-EO_SINGLE_CLASS_slicing_revised'
        s1_path = '/data/' + save_folder + '/' + file + '/s1/images/'
        s2_path = '/data/' + save_folder + '/' + file + '/s2/images/'
        s3_path = '/data/' + save_folder + '/' + file + '/s3/images/'
        l1_path = '/data/' + save_folder + '/' + file + '/s1/labels/'
        l2_path = '/data/' + save_folder + '/' + file + '/s2/labels/'
        l3_path = '/data/' + save_folder + '/' + file + '/s3/labels/'
        if not os.path.exists(s1_path):
            os.makedirs(s1_path)
            os.makedirs(s2_path)
            os.makedirs(s3_path)
            os.makedirs(l1_path)
            os.makedirs(l2_path)
            os.makedirs(l3_path)
            print('folder made!')

        if len(l1)>0:
            img = s1.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(  '/data/' + save_folder + '/' + file + '/s1/images/' + img_file_name + '.jpg', img)
            save_array_to_file(l1,'/data/' + save_folder + '/' + file + '/s1/labels/' + txt_file_name + '.txt')
        elif len(l1)==0:
            back1_path = '/data/' + save_folder + '/' + file + '/s1/background/'
            if not os.path.exists(back1_path):
                os.makedirs(back1_path)
                print('background folder made!')

            img = s1.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(back1_path + img_file_name + '.jpg', img)

        if len(l2)>0:
            img = s2.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite('/data/' + save_folder + '/' + file + '/s2/images/' + img_file_name + '.jpg', img)
            save_array_to_file(l2,'/' + save_folder + '/' + file + '/s2/labels/' + txt_file_name + '.txt')

        elif len(l2)==0:
            back2_path = '/data/' + save_folder + '/' + file + '/s2/background/'
            if not os.path.exists(back2_path):
                os.makedirs(back2_path)
                print('background folder made!')
            img = s2.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(back2_path + img_file_name + '.jpg', img)

        if len(l3)>0:
            img = s3.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite('/data/' + save_folder + '/' + file + '/s3/images/' + img_file_name + '.jpg', img)
            save_array_to_file(l3,'/data/' + save_folder + '/' + file + '/s3/labels/' + txt_file_name + '.txt')

        elif len(l3)==0:
            back3_path = '/data/' + save_folder + '/' + file + '/s3/background/'
            if not os.path.exists(back3_path):
                os.makedirs(back3_path)
                print('background folder made!')

            img = s3.detach().cpu().numpy() # tensor -> numpy
            img = np.transpose(img, (1, 2, 0)) # [C,H,W] -> [H,W,C]
            img = 255*(img) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            cv2.imwrite(back3_path + img_file_name + '.jpg', img)
