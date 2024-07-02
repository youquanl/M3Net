import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)
from petrel_client.client import Client
client = Client('~/.petreloss.conf')
import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer
import cv2
import random
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
# from torch.utils.data import Dataset
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
import copy
logger = logging.getLogger(__name__)



def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    # image_pth = '/root/OpenSeeD/demo/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297544912404.jpg'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # thing_classes = ['car','person','traffic light', 'truck', 'motorcycle']
    # stuff_classes = ['building','sky','street','tree','rock','sidewalk']

    # thing_classes = [ "car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "bicyclist", "motorcyclist"]
    # stuff_classes =[ "road", "parking", "sidewalk", "other ground", "building", "fence", "tree", "trunk", "terrain", "pole", "traffic sign","sky"]

    thing_classes = [
        "barrier", "bicycle", "bus", "car", "construction vehicle",  "motorcycle", 
        "person", "traffic cone", "trailer", "truck",
    ]
    stuff_classes =[
        "road", "other flat", "sidewalk", "terrain", "manmade", "tree", "sky",
    ]

    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)



    
    root_path = '/data/sets/nuScenes/'
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=root_path, verbose=False
    )
    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]


    sp_root = 's3://{save_root}/image_logits/nuScenes/openseed/'
    # 850
    #  0， 26， 53，80， 107， 134， 161，188， 215，242，269，296，323，
    # 350，377，404，431，458，485， 512，539，566，593，620，647，674，701，728，755，782，809，836，850
    phase_scenes = create_splits_scenes()["train"]
    # for scene_idx in tqdm(range(len(nusc.scene))[269:296]):

    with torch.no_grad():
        for scene_idx in tqdm(range(len(nusc.scene))):
            print(scene_idx)
            scene = nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                current_sample_token = scene["first_sample_token"]
                while current_sample_token != "":
                    current_sample = nusc.get("sample", current_sample_token)
                    data = current_sample["data"]
                    pointsensor = nusc.get("sample_data", data["LIDAR_TOP"])
                    pcl_path = os.path.join(nusc.dataroot, pointsensor["filename"])
                    pc_original = LidarPointCloud.from_file(pcl_path)
                    for camera_name in camera_list:
                            pc = copy.deepcopy(pc_original)
                            cam = nusc.get("sample_data", current_sample["data"][camera_name])
                            image_ori = cv2.imread(os.path.join(nusc.dataroot, cam["filename"]))
                            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                            image_ori = Image.fromarray(image_ori)
                            width = image_ori.size[0]
                            height = image_ori.size[1]
                            image = transform(image_ori)
                            image = np.asarray(image)
                            image_ori = np.asarray(image_ori)
                            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

                            batch_inputs = [{'image': images, 'height': height, 'width': width}]
                            outputs = model.forward(batch_inputs)
                            # image_logits = outputs[-1]['sem_seg'].unsqueeze(0)[:,:-1,:,:].detach().cpu().numpy()

                            image_logits = outputs[-1]['sem_seg'][:-1,:,:].permute(1,2,0).contiguous().detach().cpu().numpy()
                            # print(image_logits.shape)


                            # image_logits = map_pointcloud_to_image(nusc, current_sample["data"], image_logits)

                            cs_record = nusc.get(
                                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
                            )
                            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
                            pc.translate(np.array(cs_record["translation"]))

                            # Second step: transform from ego to the global frame.
                            poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
                            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
                            pc.translate(np.array(poserecord["translation"]))

                            # Third step: transform from global into the ego vehicle frame for the
                            # timestamp of the image.
                            poserecord = nusc.get("ego_pose", cam["ego_pose_token"])
                            pc.translate(-np.array(poserecord["translation"]))
                            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

                            # Fourth step: transform from ego into the camera.
                            cs_record = nusc.get(
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
                            min_dist = 1.0
                            mask = np.logical_and(mask, depths > min_dist)
                            mask = np.logical_and(mask, points[:, 0] > 0)
                            mask = np.logical_and(mask, points[:, 0] < image_ori.shape[1] - 1)
                            mask = np.logical_and(mask, points[:, 1] > 0)
                            mask = np.logical_and(mask, points[:, 1] < image_ori.shape[0] - 1)
                            matching_points = np.where(mask)[0]

                            matching_pixel = points[matching_points]
                            matching_pixels = np.round(
                                np.flip(matching_pixel, axis=1)
                            ).astype(np.int64)

                            # import pdb; pdb.set_trace()
                            assert len(matching_points) == len(matching_pixels)
                            image_logits = image_logits[matching_pixels[:, 0], matching_pixels[:, 1],:]
                            save_path = sp_root + cam["token"] + ".bin"
                            image_logits = image_logits.astype(np.float32)
                            client.put(save_path, image_logits.tobytes())

                    current_sample_token = current_sample["next"]
        



if __name__ == "__main__":
    main()
    sys.exit(0)