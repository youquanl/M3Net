import os
import numpy as np
from collections.abc import Sequence
import pickle
from nuscenes import NuScenes
from .builder import DATASETS
from .defaults import DefaultDataset
from pyquaternion import Quaternion
# from torch.utils.data import Dataset
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
import cv2
from PIL import Image
import copy
import torch
from petrel_client.client import Client
from copy import deepcopy
import torch.nn.functional as F
from typing import Tuple
import json
import random
client = Client('~/.petreloss.conf')

    
@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self,
                 split='train',
                 data_root='data/nuscenes',
                 sweeps=10,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1,
                 ignore_index=-1):
        self.sweeps=sweeps
        self.ignore_index = ignore_index
        self.data_root = "/data/sets/nuScenes"
        self.learning_map = self.get_learning_map(ignore_index)
        self.split = split
        if self.split == "train" or self.split == "val":
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=False)
        else:
            self.data_root = "/data/sets/nuScenes"

        
        self.thing_list = [9,14,15,16,17,18,21,2,3,4,6,12,22,23]


        super().__init__(split=split,
                         data_root=data_root,
                         transform=transform,
                         test_mode=test_mode,
                         test_cfg=test_cfg,
                         loop=loop)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            # return os.path.join(self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl")
            return  create_splits_scenes()[split] #os.path.join(self.data_root, "nuscenes_infos_train_old.pkl")
        elif split == "val":
            # return os.path.join(self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl")
            return create_splits_scenes()[split]  #os.path.join(self.data_root, "nuscenes_infos_val_old.pkl")
        elif split == "test":
            return os.path.join(self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl")
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = self.get_info_path(self.split)
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []

        if self.split == "train" or self.split == "val":
            for scene_idx in range(len(self.nusc.scene)):
                scene = self.nusc.scene[scene_idx]
                if scene["name"] in info_paths:

                    current_sample_token = scene["first_sample_token"]
                    # Loop to get all successive keyframes
                    list_data = []
                    while current_sample_token != "":
                        current_sample = self.nusc.get("sample", current_sample_token)
                        list_data.append(current_sample["data"])
                        current_sample_token = current_sample["next"]

                    # Add new scans in the list
                    data_list.extend(list_data)
        else:
            for info_path in info_paths:
                with open(info_path, 'rb') as f:
                    info = pickle.load(f)
                    data_list.extend(info)


        return data_list
    
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
    
    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points


        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        img_logits = np.empty((0, 16), dtype=np.float32)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]

        image_names = []
        images = []
        sam_features = []
        grids = []

        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            # im = image_buffer[camera_name]
            image_names.append(cam["filename"])
            image_name = cam["filename"].split('/')[-1]
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]

            matching_pixel = points[matching_points]
            # matching_pixel_org = copy.deepcopy(matching_pixel)
            matching_pixel = self.apply_coords(matching_pixel, (im.shape[0],im.shape[1]))
            # if self.split == "train":
            sam_feature_path = os.path.join("s3://{save_root}/image_embedding/nuScenes/sam/",cam["token"] + ".bin")
            sam_feature = torch.tensor(np.frombuffer(client.get(sam_feature_path), dtype=np.float32).reshape(1, 256, 64, 64))
            
            grid = torch.tensor((matching_pixel/1024)*2 - 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sam_feature = F.grid_sample(sam_feature, grid, align_corners=False).squeeze().transpose(0, 1)  # NxC #.squeeze(0).numpy()
            img_logits_path = os.path.join("s3://{save_root}/image_logits/nuScenes/openseed/",cam["token"] + ".bin")
            img_logit = np.frombuffer(client.get(img_logits_path), dtype=np.float32).reshape(-1, 16)


            matching_pixels = np.round(
                np.flip(matching_pixel, axis=1)
            ).astype(np.int64)
            images.append(im / 255)
            sam_features.append(sam_feature)
            # img_logits.append(img_logit)
            img_logits = np.concatenate((img_logits,img_logit))
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )

        return images, pairing_points, pairing_images, sam_features, img_logits

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        if self.split == "train" or self.split == "val":
            pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
            lidar_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])

            lidar_path = os.path.join(self.nusc.dataroot, lidar_path)
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            coord = points[:, :3]
            
            strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [-1, 1]

            # if "gt_segment_path" in data.keys():
            # lidar_sd_token = self.nusc.get('sample', data['token'])['data']['LIDAR_TOP']
            gt_segment_path = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', data["LIDAR_TOP"])['filename'])
            
            segment = np.fromfile(str(gt_segment_path), dtype=np.uint8, count=-1).reshape([-1])
            points_labels = copy.deepcopy(segment)
            subpath2 = gt_segment_path.split('/')[-1]
            inst_data =  np.load(os.path.join("/data/sets/nuScenes/panoptic/v1.0-trainval",subpath2.replace("lidarseg.bin", "panoptic.npz")))['data'].astype(np.int32).reshape([-1, 1])
            assert inst_data is not None
            valid = np.isin(points_labels.reshape(-1,1), self.thing_list).reshape(-1)
            
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int64)
            if self.split == "train":
                images, pairing_points, pairing_images,sam_features,img_logits  = self.map_pointcloud_to_image(data)
                images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
                sam_features = torch.cat(sam_features,0)
                img_logits = torch.tensor(img_logits)
                data_dict = dict(coord=coord, strength=strength, segment=segment, valid=valid, inst_data=inst_data,pairing_points=pairing_points, pairing_images=pairing_images, sam_features=sam_features,
                                        img_logits=img_logits,)
            elif self.split == "val":
                data_dict = dict(coord=coord, strength=strength, segment=segment, valid=valid, inst_data=inst_data)
                
        else:
            data = self.data_list[idx % len(self.data_list)]
            lidar_path = os.path.join(self.data_root, "raw",data["lidar_path"])
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            coord = points[:, :3]
            strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [-1, 1]
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
            data_dict = dict(coord=coord, strength=strength, segment=segment)
            
   
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        if self.split == "test":
            return self.data_list[idx % len(self.data_list)]["lidar_token"]
        else:
            return os.path.join(self.data_root,self.nusc.get("sample_data",self.data_list[idx % len(self.data_list)]["LIDAR_TOP"])["filename"])

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
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
        return learning_map
    @staticmethod
    def get_learning_map_kitti(ignore_index):
        learning_map_kitti = {
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
        return learning_map_kitti

    @staticmethod
    def get_learning_map_kitti_remap(ignore_index):
        learning_map_kitti_remap = {
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
            11:27,
            12:28,
            13:29,
            14:30,
            15:31,
            16:32,
            17:33,
            18:34,
        }
        return learning_map_kitti_remap

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
            # 22: 57,
        }
        return learning_map_waymo_remap

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