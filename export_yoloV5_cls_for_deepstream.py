"""
----------------------------------------------------------------------------
Filename: export_yoloV5_cls_for_deepstream.py
Path: export_yoloV5_cls_for_deepstream.py
Date: 10.06.2023
Author: youngjae.you
Purpose: Deepstream에 적용하기 위한 YOLOv5 Classification 모델을 ONNX 형식으로 변환하는 스크립트
History:
   - 
Usage:
    - python export_yoloV5_cls_for_deepstream.py --weights yolov5s-cls.pt -s 224 224 --opset 12 --simplify --dynamic
    
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation
----------------------------------------------------------------------------
"""

import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.torch_utils import select_device
from models.yolo import Detect


class ClassificationOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        scores = torch.softmax(x, dim=1)  # Apply Softmax
        return scores,


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolov5_export(weights, device):
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening Classification model\n')

    device = select_device('cpu')
    model = attempt_load(args.weights, device=device)

    model = nn.Sequential(model, ClassificationOutput())

    onnx_input_im = torch.zeros(args.batch, 3, *args.size).to(device)
    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset,
                      do_constant_folding=True, input_names=['input'], output_names=['outputs'],
                      dynamic_axes=dynamic_axes if args.dynamic else None)

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)

def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv5 Classification model conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[224], help='Inference size [H,W] (default [224])')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Implicit batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and implicit batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
