# SAM-SPT

The official implementation of our AAAI 2025 publication [SPT](https://ojs.aaai.org/index.php/AAAI/article/view/33420).

### Preparation

#### Environment

```bash
conda create -n spt python=3.8
conda activate spt
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib
pip install opencv-python
pip install timm
pip install scikit-image
pip install imgaug
pip install pandas
```

#### Datasets

Please download the dataset using [this link](https://drive.google.com/file/d/10vfrT9Su8sZcamjS3qUvN5n43Q2Y7SHm/view?usp=drive_link) and extract the archive directly into the repository root.  

#### Checkpoints

* Pre-trained SAM checkpoints.  [Download here](https://drive.google.com/file/d/1ENOoF5Nnh4K7QgB0BtwA5HQzCUeL9_qZ/view?usp=drive_link)

  ```
  pretrained_checkpoint/
  ├── vit_b.pth
  ├── vit_h.pth
  └── vit_l.pth
  ```

* SPT checkpoints.  [Download here](https://drive.google.com/file/d/1WIFAB-YvoIcSRJ_SfDPrmWuzw3mZXh3S/view?usp=drive_link)

  ```
  spt_ckpt/
  ├── spt_vit_b.pth
  ├── spt_vit_h.pth
  └── spt_vit_l.pth
  ```



### Getting Started

To launch inference in one step, simply run:

```bash
sh main.sh
```



### Citation

If you find our work helpful for your research, please consider citing:

```
@inproceedings{yang2025promptable,
  title={Promptable anomaly segmentation with sam through self-perception tuning},
  author={Yang, Hui-Yue and Chen, Hui and Wang, Ao and Chen, Kai and Lin, Zijia and Tang, Yongliang and Gao, Pengcheng and Quan, Yuming and Han, Jungong and Ding, Guiguang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13017--13025},
  year={2025}
}
```



### Acknowledgments

This codebase is built upon [SAM](https://github.com/facebookresearch/segment-anything), [LoRA](https://github.com/microsoft/LoRA),  [HQ-SAM](https://github.com/SysCV/sam-hq), [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) 

Thanks for their public code and released models.
