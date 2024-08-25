# MaskFeat3D

This repository contains the PyTorch implementation of paper: [3D Feature Prediction for Masked-AutoEncoder-Based Point Cloud Pretraining](https://openreview.net/forum?id=LokR2TTFMs).


## Installation

Our code has been tested with Ubuntu 18.04, PyTorch 1.8.1, Python 3.7, and CUDA 10.2. The reported results can be reproduced using the same environment configuration.

You can create an Anaconda environment using the following script.

```
conda create -n maskfeat3d python=3.7
conda activate maskfeat3d

# Install Pytorch
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

# Install MinkowskiEngine
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Install Extensions
cd extensions/chamfer_dist/
python setup.py install --user
cd ../emd/
python setup.py install --user
cd ../../

pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

pip install -r requirements.txt
```

## Datasets

We pretrained our model on ShapeNet with some preprocessing. Please refer to `datasets` for more details.

## Usage

### Pretrain

PointViT model example:

```
python main.py --config cfgs/pretrain_pointvit.yaml --pretrain=maskfeat

# Multi-GPU
python -m torch.distributed.launch --master_port=xxxx --nproc_per_node=8 main.py --launcher pytorch --sync_bn --config cfgs/pretrain_pointvit.yaml --pretrain=maskfeat 
```

MinkowskiNet / PointNeXt: Coming soon ...

### Downstream Task

ScanObjectNN

```
python main.py --config=cfgs/finetune_scan_hardest.yaml --exp_name=scanobjnn --ckpts=/path/to/pretrained/weights --finetune_model
```

ShapeNetPart

```
python main.py --ckpts=/path/to/pretrained/weights --root=../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300 --log_dir=shapenetpart --model=pt
```

ScanNet Detection: Coming Soon ...

### Model Weights

We provide our pretrained models, fine-tuned downstream task trained models, and the relevant training logs here: [Google Drive](https://drive.google.com/drive/folders/1uGJLK_iTQv2J48KC_YSUSlMs_fo1TO99?usp=sharing).

## Acknowledgements

We would like to thank and acknowledge referenced codes from [Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

## Citation

If you find this repository useful in your research, please cite:

```
@article{yan20233d,
  title={3d feature prediction for masked-autoencoder-based point cloud pretraining},
  author={Yan, Siming and Yang, Yuqi and Guo, Yuxiao and Pan, Hao and Wang, Peng-shuai and Tong, Xin and Liu, Yang and Huang, Qixing},
  journal={arXiv preprint arXiv:2304.06911},
  year={2023}
}
```