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


# BDD100K Annotations
# 1:  pedestrian
# 2:  rider
# 3:  car
# 4:  truck
# 5:  bus
# 6:  train
# 7:  motorcycle
# 8:  bicycle
# 9:  traffic light
# 10: traffic sign

np.random.seed(2020)

target_classes = torch.tensor([2, 3, 4, 6])

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 60, 65), (30, 50, 30), (100, 128, 100), (210, 105, 30), (220, 20, 60),
             (192, 92, 92), (255, 28, 96), (30, 30, 150), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


Bdd100k2Coco = {
    "person": "person",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bike": "bicycle",
    "motor": "motorcycle",
    "traffic light": "traffic light",
    "train": "train",
    "traffic sign": "stop sign"
}

CocoIds = {
    "person": 1,
    "car": 3,
    "bus": 6,
    "truck": 8,
    "bike": 2,
    "motor": 4,
    "traffic light": 10,
    "train": 7,
    "traffic sign": 13  # stop sign
}

def draw_ignored_regions(img, ignored_regions, color=(0, 0, 255), alpha=0.3, thickness=2):
    """
    Draw the ignored regions on the image.

    Parameters:
        img: original image (numpy array)
        ignored_regions: np.array of shape (N, 4) containing [x1, y1, x2, y2]
        color: BGR color of the boxes
        alpha: transparency for shading (0 to 1)
        thickness: border thickness
    """
    overlay = img.copy()

    for reg in ignored_regions:
        x1, y1, x2, y2 = map(int, reg)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), thickness=thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def invalid_bboxes(bboxes, ignore_polys, threshold=0.5):
    """
    Generate a boolean mask for bounding boxes that are covered by ignored regions in more than a given 'threshold' fraction of their area.
    
    Parameters:
        bboxes: np.array of shape (N, 4) containing [x, y, w, h]
        ignore_polys: list of shapely.Polygon objects representing ignored regions
        threshold: minimum overlap fraction to mark a bounding box as invalid (default 0.5)

    Returns:
        mask: boolean np.array of shape (N,), where True indicates the bounding box should be removed
    """

    mask = np.ones(len(bboxes), dtype=bool)
    maskf = np.ones(len(bboxes), dtype=float)
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        bbox_poly = box(x1, y1, x2, y2)

        # Compute the total intersection with all ignored regions.
        intersection_area = sum(bbox_poly.intersection(igr).area for igr in ignore_polys)
        overlap_fraction = intersection_area / bbox_poly.area
        maskf[i] = overlap_fraction
        if overlap_fraction >= threshold:
            mask[i] = False  # marcar como inválido

    #dropped = np.sum(mask)
    #print(f"Ignored Regions ---> marked {dropped} out of {len(bboxes)} boxes as invalid")

    return mask,maskf



    
def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, ignored_regions=None):
    # Plots one bounding box on image img
    if ignored_regions is not None:
        if is_inside_ignored_region(x, ignored_regions):
            return img  # No graficar
            
    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False, labels=None):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        if labels is None:
            color = COLORS_10[id % len(COLORS_10)]
        else:
            color = COLORS_10[labels[i]]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        if id==-1:
            img = plot_one_box([x1, y1, x2, y2], img, (0,255,0), ' ', score=score)
        else:
            img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


















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

class LoadImages:
    def __init__(self, path, img_size=(1536, 800)):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} in not a valid path.")
        
        self.files = []
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.files.append(os.path.join(root, f))
        self.files.sort()

        if len(self.files) == 0:
            raise FileNotFoundError("No images were found in the folder.")

        img_path = self.files[0]
        img = cv2.imread(img_path)
        self.seq_h = img.shape[0]
        self.seq_w = img.shape[1]
        self.count = 0
        self.frame_rate = 30
        self.width = img_size[0]
        self.height = img_size[1]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        #print(f"Se encontraron {len(self.files)} imágenes en {path}")

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration

        img_path = self.files[self.count]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cur_img, ori_img = self.init_img(img)
        return self.count, cur_img, ori_img

    def init_img(self, img):
        ori_img = img.copy()
        h, w = img.shape[:2]
        scale = self.height / min(h, w)
        if max(h, w) * scale > self.width:
            scale = self.width / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.files)
        
class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1536, 800)):
        if not os.path.isfile(path):
            raise FileExistsError
        
        self.cap = cv2.VideoCapture(path)        
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.seq_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.seq_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print('Lenth of the video: {:d} frames'.format(self.vn))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img = self.cap.read()  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
        assert img is not None, 'Failed to load frame {:d}'.format(self.count)

        cur_img, ori_img = self.init_img(img)
        return self.count, cur_img, ori_img

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.width:
            scale = self.width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return self.vn  # number of files


















class MOTR(object):
    def update(self, dt_instances: Instances):
        ret = []
        for i in range(len(dt_instances)):
            label = dt_instances.labels[i]
            if label in target_classes:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i+1]], axis=-1)
                # print("Label=",label,", bbox=", box_with_score, ", id=", id+1)
                ret.append(np.concatenate((box_with_score, [id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

class Detector(object):
    def __init__(self, eval_mode, args):

        self.args = args
        
        self.firstRun = True
        self.EvalMode = False
        
        # build model and load weights
        self.model, _, _ = build_model(args)
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.cuda()
        self.model.eval()
        
        
        if os.path.isfile(args.input_video):
            print("Source: Video")
            vid_name, prefix = args.input_video.split('/')[-1].split('.')
            self.save_root = os.path.join(args.output_dir, 'results', vid_name)
            Path(self.save_root).mkdir(parents=True, exist_ok=True)
            
            self.save_img_root = os.path.join(self.save_root, 'imgs')
            Path(self.save_img_root).mkdir(parents=True, exist_ok=True)
            
            self.txt_root = os.path.join(self.save_root, f'{vid_name}.txt')
            self.vid_root = os.path.join(self.save_root, args.input_video.split('/')[-1])
            self.dataloader = LoadVideo(args.input_video)

        elif os.path.isdir(args.input_video):  # Entrada: carpeta de imágenes
            if eval_mode:
                dir_name = os.path.basename(os.path.dirname(os.path.normpath(args.input_video)))
                self.gtloader = LoadAnnotations(args.gt_file)
                self.EvalMode = True
                seqinfo_path = os.path.join(os.path.dirname(args.input_video), 'seqinfo.ini')
                
                # Load .ini file
                config = configparser.ConfigParser()
                config.read(seqinfo_path)
                ignored_region_str = config['Sequence']['ignoredRegion']
                ignored_region_list = ast.literal_eval(ignored_region_str)
                self.ignored_regions = np.array(ignored_region_list, dtype=float)
                if len(self.ignored_regions)==0:
                    self.ignored_regions = None
                else:
                    self.ignored_regions[:,2] += self.ignored_regions[:,0]
                    self.ignored_regions[:,3] += self.ignored_regions[:,1]
                    
                    self.ignored_regions = [
                        [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        for x1, y1, x2, y2 in self.ignored_regions
                    ]
                    self.ignored_regions = [Polygon(region) for region in self.ignored_regions]
                    
                self.evaluate = False
            else:
                dir_name = os.path.basename(os.path.normpath(args.input_video))
            self.save_root = os.path.join(args.output_dir, 'results', dir_name)
            Path(self.save_root).mkdir(parents=True, exist_ok=True)
            
            self.save_img_root = os.path.join(self.save_root, 'imgs')
            Path(self.save_img_root).mkdir(parents=True, exist_ok=True)
            
            self.txt_root = os.path.join(self.save_root, f'{dir_name}.txt')
            if self.evaluate == False:
                if os.path.exists(self.txt_root):
                    os.remove(self.txt_root)
            self.vid_root = os.path.join(self.save_root, f'{dir_name}.mp4')
            if self.EvalMode:
                self.txt_root_filt = os.path.join(self.save_root, f'{dir_name}-filter.txt')
                if self.evaluate == False:
                    if os.path.exists(self.txt_root_filt):
                        os.remove(self.txt_root_filt)
            
            self.dataloader = LoadImages(args.input_video)
        else:
            raise ValueError(f"The provided path is not a valid file nor a directory: {args.input_video}")
            
        # build tracker
        self.tr_tracker = MOTR()

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def write_results(self, txt_path, frame_id, bbox_xyxy, identities, scores):
        # frame,id,x,y,w,h,mark,visibility,ref_pts.x,ref_pts.y,ref_pts.z
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},1,-1,-1,-1\n'
        with open(txt_path, 'a') as f:
            for xyxy, track_id, score in zip(bbox_xyxy, identities, scores):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(frame=int(frame_id), id=int(track_id), x1=x1, y1=y1, w=w, h=h, score=score)
                f.write(line)

    @staticmethod
    def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None, labels=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_show = img
        if ref_pts is not None:
            img_show = draw_points(img_show, ref_pts)
        if gt_boxes is not None:
            img_show = draw_bboxes(img_show, gt_boxes, identities=np.ones((len(gt_boxes), )) * -1)
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img_show, np.concatenate([dt_instances.boxes, dt_instances.scores.reshape(-1, 1)], axis=-1), dt_instances.obj_idxes, labels=labels)
        else:
            img_show = draw_bboxes(img_show, dt_instances.boxes, dt_instances.obj_idxes, labels=labels)
        cv2.imwrite(img_path, img_show)
        return img_show

    def run(self, prob_threshold=0.3, area_threshold=100, vis=True, dump=True):
        # save as video
        fps = self.dataloader.frame_rate
        videowriter = cv2.VideoWriter(self.vid_root, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.dataloader.seq_w, self.dataloader.seq_h))
        track_instances = None
        fid = 0
        if self.EvalMode:
            gtiter = iter(self.gtloader)
        try:
            for _, cur_img, ori_img in tqdm(self.dataloader):
                if cur_img is None or cur_img.numel() == 0:
                    print(f"Frame vacío en id {fid}, se detiene el loop")
                    break  # sale del for completamente
                
                if track_instances is not None:
                    track_instances.remove('boxes')
                    track_instances.remove('labels')

                res = self.model.inference_single_image(cur_img.cuda().float(), (self.dataloader.seq_h, self.dataloader.seq_w), track_instances)
                track_instances = res['track_instances']
                dt_instances = track_instances.to(torch.device('cpu'))
                #print(f"Frame {fid}, boxes: {dt_instances.boxes}, scores: {dt_instances.scores}")
                
                # filter det instances by score.
                dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
                dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)
                
                
                
                # Filter all corresponding fields based on the target_classes.
                mask = torch.isin(dt_instances.labels, target_classes)
                from copy import deepcopy
                filtered_instances = Instances(dt_instances.image_size)
                # Copy only the filtered fields.
                for field in dt_instances.get_fields():
                    data = dt_instances.get(field)
                    filtered_instances.set(field, data[mask])
                #dt_instances = filtered_instances

                # Filtering bboxes
                if self.ignored_regions is None:
                    filtered_instances_2 = filtered_instances
                else:
                    mask,maskf = invalid_bboxes(filtered_instances.boxes, self.ignored_regions, threshold=0.5)
                    from copy import deepcopy
                    filtered_instances_2 = Instances(filtered_instances.image_size)
                    # Copy only the filtered fields.
                    for field in filtered_instances.get_fields():
                        data = filtered_instances.get(field)
                        filtered_instances_2.set(field, data[mask])
                
                if vis:
                    if self.EvalMode:
                        gt_fid,gt_data = next(gtiter)
                        
                    else:
                        gt_fid = fid
                        gt_data = None
                    vis_img_path = os.path.join(self.save_img_root, '{:06d}.jpg'.format(fid+1))
                    vis_img = self.visualize_img_with_bbox(vis_img_path, ori_img, filtered_instances,gt_boxes=gt_data,labels=filtered_instances.labels)
                    # vis_img = draw_ignored_regions(vis_img, ignored_regions, color=(0, 0, 255), alpha=0.4)
                    videowriter.write(vis_img)

                if dump:
                    if self.EvalMode:
                        tracker_outputs = self.tr_tracker.update(filtered_instances_2)
                        self.write_results(txt_path=self.txt_root_filt,
                                        frame_id=(fid+1),
                                        bbox_xyxy=tracker_outputs[:, :4],
                                        identities=tracker_outputs[:, 5],
                                        scores=tracker_outputs[:, 4])
                    tracker_outputs = self.tr_tracker.update(filtered_instances)
                    self.write_results(txt_path=self.txt_root,
                                    frame_id=(fid+1),
                                    bbox_xyxy=tracker_outputs[:, :4],
                                    identities=tracker_outputs[:, 5],
                                    scores=tracker_outputs[:, 4])
                fid += 1
            videowriter.release()
        except KeyboardInterrupt:
            videowriter.release()



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
    if input_path.is_file():
        # Check for video file
        if is_video_file(input_path):
            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            print("video")
            detector = Detector(0, args)
            detector.run()

    elif input_path.is_dir():
        if contains_images(input_path):
            # Check for image sequence path
            if args.output_dir:
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            print("Images")
            detector = Detector(0,args)
            detector.run()

        else:
            if os.path.exists("output/results-motr.csv"):
                os.remove("output/results-motr.csv")
            
            # It is a root folder (containing train/, val/, and test/ subfolders, each with img/ directories).
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
                            detector = Detector(1, args)
                            detector.run()

    else:
        print(f"Invalid path: {args.input_video}")