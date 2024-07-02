import os
import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
import cv2
import copy
import torch
from petrel_client.client import Client
client = Client('~/.petreloss.conf')
from copy import deepcopy
from typing import Tuple
import torch.nn.functional as F
import json
import random
import pickle
import yaml

@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self,
                 split='train',
                 data_root='data/semantic_kitti',
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1,
                 ignore_index=-1):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_uni = self.get_learning_map_uni(ignore_index)
        self.data_root = data_root
        self.split = split
        with open("./semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        self.things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person', 'bicyclist', 'motorcyclist']
        self.stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
        self.things_ids = []
        for i in sorted(list(semkittiyaml['labels'].keys())):
            if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in self.things:
                self.things_ids.append(i)

        super().__init__(split=split,
                         data_root=data_root,
                         transform=transform,
                         test_mode=test_mode,
                         test_cfg=test_cfg,
                         loop=loop)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError
 
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, seq)
            seq_files = sorted(
                os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
        return data_list 
    
    def select_points_in_frustum(self, points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def read_calib(self, calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out
    

    
    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], 1024
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def map_pointcloud_to_image(self, ann_info, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        points = np.fromfile(ann_info, dtype=np.float32).reshape((-1, 4))
        path_splits = ann_info.split('/')
        calib_path = os.path.join(self.data_root,path_splits[-3], "calib.txt")
        image_path = os.path.join("/data/sets/semantickitti/imgages/sequences/",path_splits[-3],"image_2", path_splits[-1].replace("bin", "png"))

        image = cv2.imread(image_path)
        image = cv2.resize(image, (1241, 376), interpolation=cv2.INTER_LINEAR)

        calib = self.read_calib(calib_path)
        proj_matrix = calib['P2'] @ calib['Tr']
        proj_matrix = proj_matrix.astype(np.float32)

        # project points into image
        keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([points[:, :3], np.ones([len(points), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        matching_pixel = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(matching_pixel, 0, 0, 1241, 376)
        keep_idx = keep_idx_img_pts & keep_idx
        matching_pixel = matching_pixel[keep_idx]


        matching_pixel = self.apply_coords(matching_pixel, (image.shape[0],image.shape[1]))
        sam_feature_path = os.path.join("s3://{save_root}/image_embedding/semantickitti/sam/sequences",path_splits[-3], path_splits[-1])
        sam_feature = torch.tensor(np.frombuffer(client.get(sam_feature_path), dtype=np.float32).reshape(1, 256, 64, 64))
        grid = torch.tensor((matching_pixel/1024)*2 - 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sam_feature = F.grid_sample(sam_feature, grid, align_corners=False).squeeze().transpose(0, 1)  # NxC #.squeeze(0).numpy()
        img_logits_path =os.path.join("s3://{save_root}/image_logits/semantickitti/openseed/sequences",path_splits[-3], path_splits[-1])
        img_logit = torch.tensor(np.frombuffer(client.get(img_logits_path), dtype=np.float32).reshape(-1, 19))

        matching_pixel = np.fliplr(matching_pixel)

        pairing_points = np.where(keep_idx==True)[0]

        pairing_images = np.concatenate(
                            (
                                np.zeros((matching_pixel.shape[0], 1), dtype=np.int64),
                                matching_pixel,
                            ),
                            axis=1,
                        )

        assert pairing_images.shape[1] == 3

        images = [image / 255]
        sam_features = [sam_feature]
        img_logits = img_logit
        
        return images, pairing_points, pairing_images, sam_features,img_logits
    
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, 'rb') as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        
        strength = scan[:, -1].reshape([-1, 1])
        label_file = data_path.replace('velodyne', 'labels').replace('.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                inst_data = copy.deepcopy(segment)
                sem_labels = inst_data & 0xFFFF #delete high 16 digits binary
                valid = np.isin(sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
                segment = np.vectorize(self.learning_map.__getitem__)(segment & 0xFFFF).astype(np.int32)
                # segment = np.vectorize(self.learning_map_uni.__getitem__)(segment).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
            data_dict = dict(coord=coord, strength=strength, segment=segment)
            return data_dict
        if self.split == "train":
            images, pairing_points, pairing_images,sam_features, img_logits= self.map_pointcloud_to_image(data_path)
            images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
            sam_features = torch.cat(sam_features,axis=0)
            data_dict = dict(coord=coord, strength=strength, segment=segment, images=images, pairing_points=pairing_points, pairing_images=pairing_images,sam_features=sam_features,img_logits=img_logits, valid=valid, inst_data=inst_data)
        else:
            data_dict = dict(coord=coord, strength=strength, segment=segment, valid=valid, inst_data=inst_data)
        
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
    
    @staticmethod
    def get_learning_map_nusc(ignore_index):
        learning_map_nusc = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map_nusc

    @staticmethod
    def get_learning_map_nusc_remap(ignore_index):
        learning_map_nusc_remap = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 19, 
            1: 20, 
            2: 21, 
            3: 22, 
            4: 23, 
            5: 24, 
            6: 25, 
            7: 26, 
            8: 27, 
            9: 28, 
            10:29, 
            11: 30,
            12: 31,
            13: 32,
            14: 33,
            15: 34,
        }
        return learning_map_nusc_remap

    @staticmethod
    def get_learning_map_waymo_remap(ignore_index):
        learning_map_waymo_remap = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 35,
            1: 36, 
            2: 37, 
            3: 38, 
            4: 39, 
            5: 40, 
            6: 41, 
            7: 42, 
            8: 43, 
            9: 44, 
            10: 45, 
            11:46, 
            12: 47,
            13: 48,
            14: 49,
            15: 50,
            16: 51,
            17: 52,
            18: 53,
            19: 54,
            20: 55,
            21: 56,
        }
        return learning_map_waymo_remap

    @staticmethod
    def get_learning_map_uni(ignore_index):
        learning_map_uni = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 16, 
            1: 17, 
            2: 18, 
            3: 19, 
            4: 20, 
            5: 21, 
            6: 22, 
            7: 23, 
            8: 24, 
            9: 25, 
            10:26, 
            11: 27,
            12: 28,
            13: 29,
            14: 30,
            15: 31,
            16: 32,
            17: 33,
            18: 34,
        }
        return learning_map_uni
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)