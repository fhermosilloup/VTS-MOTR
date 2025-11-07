# VTS-MOTR
This repository contains the **source code and experimental setup** for the paper:

> **"A Transformer-Based Multi-Task Learning Model for Vehicle Traffic Surveillance"**  
> by Fernando Hermosillo-Reynoso *et al.*, 2025  

## Model Checkpoint

The pretrained model checkpoint can be downloaded from the following link:
[**Model**](https://link-al-checkpoint.com)

Once downloaded, place the checkpoint file inside the following directory:

```text
checkpoints/
```

## Datasets

### UA-DETRAC Benchmark

All experiments in this work are conducted using the **UA-DETRAC** dataset, a widely used benchmark for **multi-object detection and tracking** in vehicle traffic surveillance.

**Download Dataset:**  
- The dataset required for training and evaluation can be downloaded from the following Google Drive link:  [UA-DETRAC Dataset (Google Drive)](https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view)
- **Official Project Page:**  [UA-DETRAC Benchmark – Official Site](https://sites.google.com/view/daweidu/projects/ua-detrac?authuser=0)

### Dataset Reference

> **UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking**  
> Longyin Wen, Dawei Du, Zhaowei Cai, Zhen Lei, Ming-Ching Chang, Honggang Qi, Jongwoo Lim, Ming-Hsuan Yang, and Siwei Lyu.  
> *Computer Vision and Image Understanding (CVIU)*, 2020.  
> [DOI: 10.1016/j.cviu.2020.103048](https://doi.org/10.1016/j.cviu.2020.103048)

If you use this dataset in your research, please cite the above paper.

### Dataset Organization

After downloading and extracting the dataset, the images must be organized into the following directory structure for the model to correctly access the training and validation sequences:

```text
datasets/
└── DETRAC-MOT/
    ├── train/
    │   ├── MVI_XXXXXX/
    │   │   └── img/
    │   │       ├── img000001.jpg
    │   │       ├── img000002.jpg
    │   │       └── ...
    │   └── ...
    │
    └── val/
        ├── MVI_XXXXXX/
        │   └── img/
        │       ├── img000001.jpg
        │       ├── img000002.jpg
        │       └── ...
        └── ...
```

Make sure to **move all downloaded image sequences** into their respective folders under:

- `datasets/DETRAC-MOT/train/MVI_XXXXX/img/`  
- `datasets/DETRAC-MOT/val/MVI_XXXXX/img/`

The annotation files (bounding boxes and tracking information) are located into their corresponding video path `datasets/DETRAC-MOT/val/MVI_XXXXX/gt/gt.txt`.


## References and Code Acknowledgment

This repository is based on the implementation of **MOTR (End-to-End Multiple-Object Tracking with Transformer)** developed by **MeGVII Research**.

Original paper:
> **MOTR: End-to-End Multiple-Object Tracking with TRansformer**  
> Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.  
> *European Conference on Computer Vision (ECCV), 2022.*  
> [Paper Link (ECCV 2022)](https://arxiv.org/abs/2105.03247)

Official repository: [https://github.com/megvii-research/MOTR](https://github.com/megvii-research/MOTR)

Our Transformer-based multi-task model extends and adapts the **MOTR** framework to vehicle traffic surveillance scenarios, integrating additional heads for **vehicle classification** and **occlusion detection**, and optimizing the architecture for multi-view VTS channels.

If you use this repository in your research, please consider citing both works.

### Citation

```bibtex
@inproceedings{zheng2022motr,
  title     = {MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author    = {Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
