# VTS-MOTR
This repository contains the **source code and experimental setup** for the paper:

> **"A Transformer-Based Multi-Task Learning Model for Vehicle Traffic Surveillance"**  
> by Fernando Hermosillo-Reynoso *et al.*, 2025  

## Usage
### Installation (from [MOTR/README.md](https://github.com/megvii-research/MOTR/blob/motr_bdd100k/README.md))

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

#### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
    Specifically, in our setup:
    ```pgsql
    torch                             2.8.0+cu126
    torchaudio                        2.8.0
    torchvision                       0.23.0+cu126
    tensorboard                       2.20.0
    tensorboard-data-server           0.7.2
    numpy                             2.1.3
    detectron2                        0.6
    ```

* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```
    
### Model Checkpoint

The pretrained model checkpoint can be downloaded from the following link:
[**Model**](https://drive.google.com/file/d/1xwiUZqFWHZt2kIhCeRbSgCA0ygbolDxa/view?usp=sharing)

Once downloaded, place the checkpoint file inside the following directory:

```text
checkpoints/
```
    
### Datasets

#### UA-DETRAC Benchmark

All experiments in this work are conducted using the **UA-DETRAC** dataset, a widely used benchmark for **multi-object detection and tracking** in vehicle traffic surveillance.

**Download Dataset:**  
- The dataset required for training and evaluation can be downloaded from the following Google Drive link:  [UA-DETRAC Dataset](https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view)
- **Official Project Page:**  [UA-DETRAC Benchmark – Official Site](https://sites.google.com/view/daweidu/projects/ua-detrac?authuser=0)

#### Dataset Organization

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

### Running inference
To run inference, please execute the following command:
```bash
python demo.py --meta_arch motr --dataset_file bdd100k_mot --pretrained checkpoints/motr.pth --resume checkpoints/motr.pth --output_dir output --input_video datasets/DETRAC-MOT --batch_size 1 --sample_mode 'random_interval' --sample_interval 10 --sampler_steps 50 90 120 --sampler_lengths 2 3 4 5 --update_query_pos --merger_dropout 0 --dropout 0 --random_drop 0.1 --fp_ratio 0.3 --extra_track_attn --track_embedding_layer AttentionMergerV4 --with_box_refine
```
This will generate subfolders with the predictions of each video in the DETRACT dataset on directory `output/results/train/MVI_XXXXX`

### Running Evaluation
Before running evaluation, you must first run inference to generate the necessary prediction files.
Once inference is completed, you can run evaluation using the following command:
```bash
python eval.py --meta_arch motr --dataset_file bdd100k_mot --pretrained checkpoints/motr.pth --resume checkpoints/motr.pth --output_dir output --input_video datasets/DETRAC-MOT --batch_size 1 --sample_mode 'random_interval' --sample_interval 10 --sampler_steps 50 90 120 --sampler_lengths 2 3 4 5 --update_query_pos --merger_dropout 0 --dropout 0 --random_drop 0.1 --fp_ratio 0.3 --extra_track_attn --track_embedding_layer AttentionMergerV4 --with_box_refine

```
This will generate a `results-motr.csv` file on `output` directory.

### Training the model
To train the model, simply run:
```bash
sh configs/r50_motr_train.sh
```
After training the checkpoint will be generated at `exps\r50.ua_detrac_mot`

## References

This repository is based on the implementation of **MOTR (End-to-End Multiple-Object Tracking with Transformer)** developed by **MeGVII Research**.

Original paper:
> **MOTR: End-to-End Multiple-Object Tracking with TRansformer**  
> Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.  
> *European Conference on Computer Vision (ECCV), 2022.*  
> [Paper Link (ECCV 2022)](https://arxiv.org/abs/2105.03247)

Official repository: [MOTR](https://github.com/megvii-research/MOTR)

Our Transformer-based multi-task model extends the **MOTR** framework to vehicle traffic surveillance scenarios.

If you use this repository in your research, please consider citing both works.

### Citation

```bibtex
@inproceedings{zheng2022motr,
  title     = {MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author    = {Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}

@article{wen2020uadetrac,
  title     = {UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking},
  author    = {Longyin Wen and Dawei Du and Zhaowei Cai and Zhen Lei and Ming-Ching Chang and Honggang Qi and Jongwoo Lim and Ming-Hsuan Yang and Siwei Lyu},
  journal   = {Computer Vision and Image Understanding (CVIU)},
  year      = {2020},
  doi       = {10.1016/j.cviu.2020.103048}
}
```
