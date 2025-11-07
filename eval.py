# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from __future__ import print_function
import glob
import math
import sys
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import torchvision.transforms.functional as F
import cv2
import numpy as np
import torch
import argparse
import torchvision.transforms.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from util.evaluation import Evaluator
import motmetrics as mm
from typing import List
import shutil
from models.structures import Instances
from shapely.geometry import Polygon, box
import configparser
import ast

np.random.seed(2020)


class LoadAnnotations:
    """
    Iterator that goes through annotations in the gt.txt file frame by frame.
    Returns the detections (bbox + id) associated with each frame.
    Expected line format: frame, id, x, y, w, h, conf, x3, y3, z3, vis
    """
    def __init__(self, gt_file):
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"No se encontró el archivo: {gt_file}")
        
        # Load annotations file
        self.gt = np.loadtxt(gt_file, delimiter=",", dtype=float)
        
        # Get unique frames
        self.frames = np.unique(self.gt[:, 0]).astype(int)
        self.count = 0
        self.fid = 0

    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        self.fid += 1
        if self.count >= len(self.frames):
            print("CurFrame:",self.fid)
            print("LastFrame:",self.frames[-1])
            raise StopIteration

        frame_id = self.frames[self.count]
        if frame_id != self.fid:
            self.count -= 1
            return frame_id,None
        
        # Extract annotations for frame==frame_id
        ann = self.gt[self.gt[:, 0] == frame_id]

        # Get columns: x, y, w, h, id
        if ann.size > 0:
            gt_frame = ann[:, [2, 3, 4, 5]].copy()
            gt_frame[:, 2] += gt_frame[:, 0]  # x2 = x + w
            gt_frame[:, 3] += gt_frame[:, 1]  # y2 = y + h
            #frame,id,x,y,w,h,mark,label,visibility
        else:
            gt_frame = np.empty((0, 4))
        
        return frame_id,gt_frame

    def __len__(self):
        return len(self.frames)


class DetractEval(object):
    def __init__(self, args):
        self.args = args
        
        if os.path.isdir(args.input_video):
            dir_name = os.path.basename(os.path.dirname(os.path.normpath(args.input_video)))
            self.gtloader = LoadAnnotations(args.gt_file)
            self.EvalMode = True
            seqinfo_path = os.path.join(os.path.dirname(args.input_video), 'seqinfo.ini')
            
            # GT.BBOX[X,Y,W,H]
            # PRED.BBOX[X,Y,W,H]
            self.evaluator = Evaluator(os.path.dirname(os.path.dirname(args.input_video)), dir_name, data_type='mot')
            # ##frame,id,x,y,w,h,score,vis,ref_x,ref_y,ref_z
            acc=self.evaluator.eval_file(os.path.join(os.path.join(args.output_dir, 'results', dir_name), f'{dir_name}-filter.txt'))
            print("Summary:")
            summary = Evaluator.get_summary(
                [acc],
                [dir_name]
            )
            print(summary)
            fila_df=summary.loc[summary.index[0]].to_frame().T
            xfilename = "output/results-motr.csv"
            if not os.path.exists(xfilename):
                fila_df.to_csv(xfilename, index=True, header=True)
            else:
                fila_df.to_csv(xfilename, index=True, header=False, mode='a')
            results_filename = os.path.join(os.path.join(args.output_dir, 'results', dir_name), f'results_{dir_name}.xlsx')
            print("Summary.saveTo:",results_filename)
            Evaluator.save_summary(summary, results_filename)
        else:
            raise ValueError(f"The provided path is not a valid ua-detrac directory: {args.input_video}")


def is_video_file(path):
    VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv']
    return any(path.lower().endswith(ext) for ext in VIDEO_EXTS)
    
def is_image_file(filename):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    return filename.lower().endswith(exts)

def contains_images(path):
    return any(is_image_file(f) for f in os.listdir(path))

def get_video_folders(root_dir):
    img_dirs = []
    for subdir, dirs, files in os.walk(root_dir):
        if os.path.basename(subdir) == 'img' and contains_images(subdir):
            img_dirs.append(subdir)
    return img_dirs
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Add new field
    args.gt_file = None
    
    input_path = Path(args.input_video)
    
    # Check input path
    if input_path.is_dir():
        if contains_images(input_path):
            raise ValueError(f"The provided path is not a valid ua-detrac directory: {args.input_video}")
            
        else:
            # It is a root folder (containing train/, val/, and test/ subfolders, each with img/ directories).
            if os.path.exists("output/results-motr.csv"):
                os.remove("output/results-motr.csv")
            
            subfolders = [f.name for f in input_path.iterdir() if f.is_dir()]
            valid_sets = {"train", "val", "test"}
            existing_sets = [s for s in subfolders if s in valid_sets]
            if not existing_sets:
                sys.exit(f"Folders'train' / 'val' / 'test' were not found in {input_path}")
            else:
                output_dir = args.output_dir
                for subset in existing_sets:
                    subset_path = input_path / subset

                    # Search for all video folders inside the train/, val/, and test/ directories.
                    for video_folder in subset_path.iterdir():
                        gt_path = video_folder / "gt" / "gt.txt"
                        if gt_path.exists():
                            args.gt_file = str(gt_path)
                        
                        img_dir = video_folder / "img"
                        if img_dir.exists() and any(img_dir.glob("*.jpg")):
                            args.output_dir = output_dir
                            args.input_video = str(img_dir)
                            if args.output_dir:
                                # Set up the paths for this video
                                output_path = output_dir + "/" + subset
                                args.output_dir = output_path
                                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                            tmp = DetractEval(args)

    else:
        raise ValueError(f"The provided path is not a valid ua-detrac directory: {args.input_video}")