## Installation
```bash
# Installation refer https://github.com/Pointcept/Pointcept/
conda create -n m3net python=3.8 -y
conda activate m3net
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```
## Dataset Preparation
### Overall Structure

```
└── data 
    └── sets
        │── nuScenes
        │── semantickitti
        └── waymo

```

## nuScenes

To install the [nuScenes-lidarseg](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download.

Unpack the compressed file(s) into `/data/sets/nuScenes` and your folder structure should end up looking like this:

```
└── nuScenes  
    ├── panoptic (groudtruth for panoptic segmentation, each lidar point has a semantic label and an instance id)
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```

Please note that you should cite the corresponding paper(s) once you use the dataset.

```bibtex
@article{fong2022panopticnuscenes,
    author = {Whye Kit Fong and Rohit Mohan and Juana Valeria Hurtado and Lubing Zhou and Holger Caesar and Oscar Beijbom and Abhinav Valada},
    title = {Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
    journal = {IEEE Robotics and Automation Letters},
    volume = {7},
    number = {2},
    pages = {3795--3802},
    year = {2022}
}
```
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {Holger Caesar and Varun Bankiti and Alex H Lang and Sourabh Vora and Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and Giancarlo Baldan and Oscar Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```


## SemanticKITTI

To install the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset and additionally the [color data](https://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) website. 

Unpack the compressed file(s) into `/data/sets/semantickitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── semantickitti  
    └── images
    └── sequences
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 10
        ├── calib
        │    └── 00
        │    └── ···
        │    └── 21
        └── semantic-kitti.yaml
        
```

Please note that you should cite the corresponding paper(s) once you use the dataset.


```bibtex
@inproceedings{behley2019semantickitti,
    author = {Jens Behley and Martin Garbade and Andres Milioto and Jan Quenzel and Sven Behnke and Jürgen Gall and Cyrill Stachniss},
    title = {SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
    booktitle = {IEEE/CVF International Conference on Computer Vision},
    pages = {9297--9307},
    year = {2019}
}
```
```bibtex
@inproceedings{geiger2012kitti,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```



## Waymo Open
To install the [Waymo Open](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf) dataset, download the annotations from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_2/archived_files?authuser=1&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). Note that you only need to download data from training and testing prefix. Unpack the compressed file(s) into `./data/sets/waymo/raw_data/` and re-organize the data structure. Besides, the aligned images could be obtained from [here](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F), the pre-process steps, please kindly refer to [LoGoNet](https://github.com/PJLab-ADG/LoGoNet/blob/main/docs/DATA_PREPROCESS.md) for more details.

Installation. Note that we only test it with python 3.6.
```
rm -rf waymo-od > /dev/null
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od && git branch -a
git checkout remotes/origin/master
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-6-0==1.4.3
```

Next, execute the following script:
```shell
python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py
```

Lastly, download files from [`train-0-31.txt`](https://www.dropbox.com/s/ijnxe9skn3r8dbg/train-0-31.txt?dl=0) and [`val-0-7.txt`](https://www.dropbox.com/s/cqcm9mftidik0fu/val-0-7.txt?dl=0), and put them into `waymo` folder.

Your folder structure should end up looking like this:
```
└── waymo
    └── raw_data
        └── training
        └── validation
        └── testing
    └── train
        └── first
        └── second
    └── val_with_label
        └── first
        └── second
    └── train-0-31.txt
    └── val-0-7.txt
    └── images
```


Please note that you should cite the corresponding paper(s) once you use the dataset.

```bibtex
@inproceedings{sun2020waymoopen,
    author = {Pei Sun and Henrik Kretzschmar and Xerxes Dotiwalla and Aurelien Chouard and Vijaysai Patnaik and Paul Tsui and James Guo and Yin Zhou and Yuning Chai and Benjamin Caine and Vijay Vasudevan and Wei Han and Jiquan Ngiam and Hang Zhao and Aleksei Timofeev and Scott Ettinger and Maxim Krivokon and Amy Gao and Aditya Joshi and Yu Zhang and Jonathon Shlens and Zhifeng Chen and Dragomir Anguelov},
    title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2446--2454},
    year = {2020}
}
```

## SAM Features Preparation
Please refer to the following steps to generate [SAM](https://ai.meta.com/research/publications/segment-anything/) features offline.
```bash
#  For SAM installation, please kindly refer to https://github.com/facebookresearch/segment-anything
cd generation/SAM
sh sam_feature_generation_semantickitti.sh
sh sam_feature_generation_nusc.sh
sh sam_feature_generation_waymo.sh
```

## Image Logits Generation
The vision-language model (VLM) we adopted in our paper is [OpenSeeD](https://arxiv.org/pdf/2303.08131), which is trained on text-driven image alignment. Please refer to the following steps to prepare the image logits output from OpenSeeD for the SemanticKITTI, nuScenes, and Waymo Open datasets.
```bash
# For OpenSeeD installation, please kindly refer to https://github.com/IDEA-Research/OpenSeeD
cd generation/OpenSeeD/demo
sh demo_logits_generation_semantickitti.sh
sh demo_logits_generation_nusc.sh
sh demo_logits_generation_waymo.sh
```

## Training and Inference of the M3Net Model
For training the M3Net model, the configurations are defined in `configs`, and the training from scratch can be started with:
```bash
sh scripts/slurm_create.sh  -g 8 -d semantickitti -c semseg-m3net-unet-sk-nu-wa-all-level-v0  -n semseg-m3net-unet-sk-nu-wa-all-level-v0
```
For the inference:
```bash
sh scripts/slurm_create.sh  -g 8 -t test -d semantickitti -c semseg-m3net-unet-sk-nu-wa-all-level-v0  -n semseg-m3net-unet-sk-nu-wa-all-level-v0
```