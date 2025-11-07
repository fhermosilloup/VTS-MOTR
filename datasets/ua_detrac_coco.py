from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class UADetracDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        """
        img_folder: carpeta raíz con imágenes
        ann_file: JSON con anotaciones en formato UA-DETRAC (COCO-like)
        transforms: transformaciones a aplicar a las imágenes (T.Compose)
        """
        self.img_folder = img_folder
        self.transforms = transforms

        # Cargar JSON
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Diccionarios de imágenes y anotaciones
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Lista ordenada de IDs de imagen
        self.ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_folder, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        ann_list = self.annotations.get(img_id, [])

        # Procesar bboxes
        boxes = []
        labels = []
        obj_ids = []
        video_index = []
        iscrowd = []
        for a in ann_list:
            x, y, bw, bh = a["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            #boxes.append([x1, y1, x2, y2])
            boxes.append(a["bbox"])
            labels.append(a["category_id"]-1)
            obj_idx_offset = a["video_index"] * 2000
            obj_ids.append(a["Obj_id"]+obj_idx_offset)
            iscrowd.append(a["iscrowd"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        obj_ids = torch.tensor(obj_ids, dtype=torch.int64)
        video_index = torch.tensor(video_index, dtype=torch.int64)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "obj_ids": obj_ids,
            "iscrowd": iscrowd,
            "img_id": [img_id],
            "size": torch.tensor([h, w]),
            "orig_size": torch.tensor([h, w]),
            "file_name": img_info["file_name"]
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

def make_ua_detrac_transformation():
    return T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])
# ------------------------------------------------------------
# Builder para el dataset
# ------------------------------------------------------------
def build_ua_detrac(image_set):
    """
    image_set: "train" o "val"
    img_folder: ruta a las imágenes
    ann_file: JSON de anotaciones
    transforms: transformaciones a aplicar
    """
    img_folder = "datasets\\UA-DETRACT\\"
    if image_set == 'train':
        ann_file = "datasets\\UA-DETRACT\\ua_detrac_train.json"
    elif image_set == 'val':
        ann_file = "datasets\\UA-DETRACT\\ua_detrac_val.json"
    else:
        raise ValueError(f'image_set {image_set} not supported')
    dataset = UADetracDetection(img_folder, ann_file, transforms=None)
    return dataset