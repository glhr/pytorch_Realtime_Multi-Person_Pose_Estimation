#!/usr/bin/env python3
import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model('vgg19')
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).to(device)
model.float()
model.eval()

import argparse
from vision_utils.timing import CodeTimer
from vision_utils.logger import get_logger
import glob
import json

logger=get_logger()

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/robotlab/pose test input",
                    help='directory of PNG images to run fastpose on')

args = parser.parse_args()
times = []
scale=1

for test_image in glob.glob(f"{args.input_dir}/*.png"):
    img_name = f'{test_image.split("/")[-1].split(".")[-2]}-{scale}.{test_image.split(".")[-1]}' if scale<1 else test_image.split("/")[-1]

    oriImg = cv2.imread(test_image) # B,G,R order
    dim = (int(oriImg.shape[1] * scale), int(oriImg.shape[0] * scale))
    oriImg = cv2.resize(oriImg, dim)
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of original image

    with CodeTimer() as timer:
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

        print(im_scale)
        humans = paf_to_pose_cpp(heatmap, paf, cfg)

    print(img_name, timer.took)
    times.append(timer.took)


    logger.debug(humans)
    image_h, image_w = oriImg.shape[:2]
    centers = {}
    json_out = []
    for human in humans:
        # draw point
        keypoints = []
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            keypoints.extend([center[0],center[1],body_part.score])
        logger.debug(keypoints)
        json_out.append({'keypoints':keypoints})
    json_out_name = '../eval/pytorch_Realtime_Multi-Person_Pose_Estimation/' + img_name + '.predictions.json'
    with open(json_out_name, 'w') as f:
        json.dump(json_out, f)
    logger.info(json_out_name)

    out = draw_humans(oriImg, humans)
    cv2.imwrite(img_name,out)

print(np.mean(times))
