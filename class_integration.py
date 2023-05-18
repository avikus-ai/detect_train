##################### change class index ##########################
import subprocess
import torch
import torchvision
from torchvision import transforms
import PIL 
from torchvision.datasets.folder import pil_loader
import torch.nn as nn
import random
import glob
import contextlib
import hashlib
import json
import math
import os
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
import io
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import matplotlib.patches as patches
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
# from mat4py import loadmat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def get_text_file(file_path):
    """
    This function takes a file path as input and returns the contents of the file if it is a text file.
    """
    # check if the file exists
    if os.path.exists(file_path):
        # check if the file is a text file
        if file_path.endswith('.txt'):
            # open the file in read mode
            with open(file_path, 'r') as file:
                # read the contents of the file
                file_contents = file.read()
                # return the contents of the file
                return file_contents
        else:
            # if the file is not a text file, print an error message
            print('Error: File is not a text file.')
    else:
        # if the file does not exist, print an error message
        print('Error: File does not exist.')
    
# Define a function that takes a 2D array as input
def remove_zero_first(arr,del_val):
    # Create an empty list to store the non-zero first element arrays
    new_arr = []
    # Loop through each 1D array in the 2D array
    for sub_arr in arr:
        # Check if the first element of the 1D array is not equal to 0
        if sub_arr[0] != del_val:
            # If the first element is not 0, append the 1D array to the new array
            new_arr.append(sub_arr)
    # Return the new array
    return np.array(new_arr)

folder_path = '/data/batch01_new/SINGLE_CLASS/'
sub_folder_list = ['2939','ccnuri','GORAE/221020','GORAE/221026','GORAE/221028', 'hannara/busan','hannara/incheon_arriving', 'hannara/incheon_berth',\
                   'KHOPE', 'maersk/2023-01-15_10/eo1','maersk/2023-01-15_10/eo2','maersk/2023-01-15_10/eo3','maersk/2023-01-15_11/eo1','maersk/2023-01-15_11/eo2','maersk/2023-01-15_11/eo3','YMwinner']
# print(sub_folder_list)

for _ in sub_folder_list:
    train_val_list = ['train', 'val']

    for train_val in train_val_list:
        txt_dataset_folder = sorted(glob.glob(sub_folder_list + "/labels/" + train_val + "/*.txt"))
        print(len(txt_dataset_folder))

        for txt in txt_dataset_folder:
            with open(txt) as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
            lb2 = get_text_file(txt)

            # modifying to single class
            for q in range(len(lb)):
                if lb[q][0] == 1: # vessel
                    lb[q][0] = 0
                elif lb[q][0] == 8: # small vessel
                    lb[q][0] = 0
            lb = remove_zero_first(lb,2)
            lb = remove_zero_first(lb,3)
            lb = remove_zero_first(lb,4)
            lb = remove_zero_first(lb,5)
            lb = remove_zero_first(lb,6)
            lb = remove_zero_first(lb,7)

            # save modified label
            save_array_to_file(lb,txt)
