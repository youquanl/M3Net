import os
import numpy as np
import glob
import pickle
from petrel_client.client import Client
client = Client('~/.petreloss.conf')
import copy
import io
import cv2
import random
from tqdm import tqdm
import torch
from PIL import Image
from .builder import DATASETS
from .defaults import DefaultDataset
from copy import deepcopy
from typing import Tuple
import torch.nn.functional as F
import json
import random
@DATASETS.register_module()
class WaymoDataset(DefaultDataset):
    def __init__(self,
                 split='training',
                 data_root='data/waymo',
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1,
                 ignore_index=-1):
        self.ignore_index = ignore_index
        self.learning_map_uni = self.get_learning_map_uni(ignore_index)
        calib_info = '/data/sets/waymo/images_infos_new.pkl'
        self.image_root = "/data/sets/waymo/images/"
        df = open(calib_info,'rb')
        self.data = pickle.load(df)

        
                
        super().__init__(split=split,
                         data_root=data_root,
                         transform=transform,
                         test_mode=test_mode,
                         test_cfg=test_cfg,
                         loop=loop)

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]

        data_list = []
        for split in self.split:
            if "train" in split:
                with open('/data/sets/waymo/train-0-31.txt', 'r') as f:
                    for line in f.readlines():
                        data_list.append(line.strip())

            else:
                with open('/data/sets/waymo/val-0-7.txt', 'r') as f:
                    for line in f.readlines():
                        data_list.append(line.strip())
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
    
    def map_pointcloud_to_image(self, pc,ann_info):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. 

        """
        image_infos = self.data[ann_info.split('/')[-1]]
        image_info = image_infos['image']
        sample_idx = image_infos['sample_idx']
        sequence_name = image_infos['sequence_name']

        pairing_points = np.empty(0, dtype=np.int32)
        pairing_images = np.empty((0, 3), dtype=np.int32)
        img_logits = np.empty((0, 22), dtype=np.float32)
        
        images = []
        sam_features = []
        grids = []
        for i in range(5):
            points = copy.deepcopy(pc)
            image_shape_i = image_info['image_shape_{}'.format(i)]
            axes_tf = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]])
            # get the camera related parameters
            new_ex_param = np.matmul(axes_tf, np.linalg.inv(image_info['image_{}_extrinsic'.format(i)]))
            image_extrinsic_i = new_ex_param
            image_intrinsic_i = image_info['image_{}_intrinsic'.format(i)]
            image_id = image_info['image_'+str(sample_idx)+'_path'].split('/')[-1]
            image_path_i =  self.image_root +sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id
            im = np.array(Image.open(image_path_i))


            proj_matrix = image_intrinsic_i @ image_extrinsic_i
            proj_matrix = proj_matrix.astype(np.float32)
                    # project points into image
            points_hcoords = np.concatenate([points[:, :3], np.ones([len(points), 1], dtype=np.float32)], axis=1)
            img_points = (proj_matrix @ points_hcoords.T).T
            matching_pixel = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
            keep_idx_img_pts = self.select_points_in_frustum(matching_pixel, 0, 0, image_shape_i[1], image_shape_i[0]) & (img_points[:, 2] > 0)
            # print(keep_idx)
            keep_idx = keep_idx_img_pts 
            matching_pixel = matching_pixel[keep_idx]

            matching_points = np.where(keep_idx==True)[0]
            matching_pixel = self.apply_coords(matching_pixel, (im.shape[0],im.shape[1]))

            subpath = sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id.replace('png','bin')
            subpath1 = sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id.replace('jpg','bin')

            sam_feature_path = os.path.join("s3://{save_root}/image_embedding/waymo/sam/",subpath)
            sam_feature = torch.tensor(np.frombuffer(client.get(sam_feature_path), dtype=np.float32).reshape(1, 256, 64, 64))
            grid = torch.tensor((matching_pixel/1024)*2 - 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # if not self.fea_l:
            sam_feature = F.grid_sample(sam_feature, grid, align_corners=False).squeeze().transpose(0, 1)  # NxC #.squeeze(0).numpy()

            img_logits_path = os.path.join("s3://{save_root}/image_logits/waymo/openseed/",subpath1)
            img_logit = np.frombuffer(client.get(img_logits_path), dtype=np.float32).reshape(-1, 22)
            if len(img_logit) != len(sam_feature):
                print(img_logits_path)


            matching_pixels = np.fliplr(matching_pixel).astype(np.int64)
            images.append(im / 255)
            sam_features.append(sam_feature)
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
        data_path = self.data_list[idx % len(self.data_list)]

        ann_info = data_path.copy()
        raw_xyz = np.load(io.BytesIO(client.get(ann_info))).reshape(-1,7)[:,3:6].reshape((-1,3)).astype(np.float32)
        intenel = np.load(io.BytesIO(client.get(ann_info))).reshape(-1,7)[:,1:3].reshape((-1,2)).astype(np.float32)
        pc_first = np.concatenate((raw_xyz,intenel),1)

        sec_path = ann_info.replace('first/', 'second/')
        raw_xyz1 = np.load(io.BytesIO(client.get(sec_path))).reshape(-1,7)[:, 3:6].reshape((-1, 3)).astype(np.float32)
        intenel1 = np.load(io.BytesIO(client.get(sec_path))).reshape(-1,7)[:, 1:3].reshape((-1, 2)).astype(np.float32)
        pc_second = np.concatenate((raw_xyz1, intenel1), 1)

        scan = np.concatenate((pc_first, pc_second), 0).astype(np.float32).copy()

        coord = scan[:, :3]
        strength = np.tanh(scan[:, 3].reshape([-1, 1]))

        annotated_data_first = np.load(io.BytesIO(client.get(ann_info))).reshape(-1,7)[:,-1].reshape((-1,1)).astype(np.int32)
        annotated_data_second =  np.load(io.BytesIO(client.get(sec_path))).reshape(-1,7)[:, -1].reshape((-1, 1)).astype(np.int32)
        segment = np.concatenate((annotated_data_first,annotated_data_second), 0).reshape(-1) - 1  # ignore_index 0 -> -1 

        if 'train' in ann_info:
            images, pairing_points, pairing_images,sam_features,img_logits   = self.map_pointcloud_to_image(scan, ann_info)
        else:
            return dict(coord=coord, strength=strength, segment=segment)
        # images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
        sam_features = torch.cat(sam_features,0)
        img_logits = torch.tensor(img_logits)
        # # else:
        data_dict = dict(coord=coord, strength=strength, segment=segment, pairing_points=pairing_points, pairing_images=pairing_images, sam_features=sam_features, img_logits=img_logits)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    # @staticmethod
    # def get_learning_map_uni_(ignore_index):
    #     learning_map_uni = {
    #         ignore_index: ignore_index,  # "unlabeled"
    #         0: 35,  
    #         1: 36,  
    #         2: 37,  
    #         3: 38,  
    #         4: 39,  
    #         5: 40,  
    #         6: 41,  
    #         7: 42,  
    #         8: 43,  
    #         9: 44,  
    #         10:45,  
    #         11: 46, 
    #         12: 47, 
    #         13: 48, 
    #         14: 49, 
    #         15: 50, 
    #         16: 51, 
    #         17: 52, 
    #         18: 53, 
    #         19: 54, 
    #         20: 55, 
    #         21: 56, 
    #     }
    #     return learning_map_uni
    
    @staticmethod
    def get_learning_map_uni(ignore_index):
        learning_map_uni = {
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
            11:30, 
            12:31, 
            13:32, 
            14:33, 
            15:34, 
            16:35, 
            17:36, 
            18:37, 
            19:38, 
            20:39, 
            21:40, 
        }
        return learning_map_uni

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
            0: 22, 
            1: 23, 
            2: 24, 
            3: 25, 
            4: 26, 
            5: 27, 
            6: 28, 
            7: 29, 
            8: 30, 
            9: 31, 
            10:32, 
            11:33,
            12:34,
            13:35,
            14:36,
            15:37,
            16:38,
            17:39,
            18:40,
        }
        return learning_map_kitti_remap


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
            0: 41, 
            1: 42, 
            2: 43, 
            3: 44, 
            4: 45, 
            5: 46, 
            6: 47, 
            7: 48, 
            8: 49, 
            9: 50, 
            10: 51, 
            11: 52,
            12: 53,
            13: 54,
            14: 55,
            15: 56,
        }
        return learning_map_nusc_remap
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