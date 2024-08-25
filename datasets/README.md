## Dataset

The overall directory structure should be:
```
│MaskFeat3D/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ScanObjectNN/
│   ├──ShapeNetFeat/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNetPart Dataset:
```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```
Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 

### ShapeNetFeat Dataset:
```
│ShapeNetFeat/
├──pointcloud/
│  ├── 02691156
│  │  ├── 1a04e3eab45ca15dd86060f189eb133.npy
│  │  ├── ...
│  ├── 02747177
│  ├── .......
├──sv_k128/
│  ├── 02691156
│  │  ├── 1a04e3eab45ca15dd86060f189eb133.npy
│  │  ├── ...
│  ├── 02747177
│  ├── .......
├── train.txt
├── test.txt
```

ShapeNetFeat is the main dataset used for model pretraining, derived from the ShapeNet dataset. Within the `pointcloud` folder, the dataset contains point cloud data of 3D shapes from ShapeNet. Each point cloud consists of 50,000 points sampled from the 3D shape with surface normals, represented as [50000, 6]. The `sv_k128` folder contains the surface variation data for each 3D shape, calculated with a computation radius of 128 points. Currently, we provide preprocessing scripts for this dataset, and the complete dataset will be released in the future.

To generate the surface variation data, first organize the ShapeNet point cloud data as described above. Then, run the script `python utils/preprocess_shapenet.py` to compute the surface variation for each data point. Please refer to the script for further details.

