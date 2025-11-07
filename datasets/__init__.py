# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .bdd100k_mot import build as build_bdd100k_mot
from .torchvision_datasets import CocoDetection
from .ua_detrac_coco import build_ua_detrac
from .ua_detrac_mot import build_detrac_mot
from detectron2.data import DatasetCatalog, MetadataCatalog



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'bdd100k_mot':
        return build_bdd100k_mot(image_set, args)
    if args.dataset_file == 'ua_detrac_coco':
        return build_ua_detrac(image_set)
    if args.dataset_file == 'detrac_mot':
        return build_detrac_mot(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

