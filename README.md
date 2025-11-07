# VTS-MOTR
This repository contains the **source code and experimental setup** for the paper:

> **"A Transformer-Based Multi-Task Learning Model for Vehicle Traffic Surveillance"**  
> by Fernando Hermosillo-Reynoso *et al.*, 2025  

## References and Code Acknowledgment

This repository is based on the implementation of **MOTR (End-to-End Multiple-Object Tracking with Transformer)** developed by **MeGVII Research**.

Original paper:
> **MOTR: End-to-End Multiple-Object Tracking with TRansformer**  
> Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao.  
> *European Conference on Computer Vision (ECCV), 2022.*  
> [Paper Link (ECCV 2022)](https://arxiv.org/abs/2105.03247)

Official repository:  
🔗 [https://github.com/megvii-research/MOTR](https://github.com/megvii-research/MOTR)

Our Transformer-based multi-task model extends and adapts the **MOTR** framework to vehicle traffic surveillance scenarios, integrating additional heads for **vehicle classification** and **occlusion detection**, and optimizing the architecture for multi-view VTS channels.

If you use this repository in your research, please consider citing both works.

---

### Citation

```bibtex
@inproceedings{zheng2022motr,
  title     = {MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author    = {Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
