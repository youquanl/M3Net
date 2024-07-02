import numpy as np
import torch
from petrel_client.client import Client
client = Client('~/.petreloss.conf')
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def compute_sam(cam_token, sp_root):
    cam = nusc.get("sample_data", cam_token)
    image = cv2.imread(os.path.join(nusc.dataroot, cam["filename"]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # masks = mask_generator.generate(image)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().detach().cpu().numpy()

    save_path = sp_root + cam["token"] + ".bin"
    image_embedding = image_embedding.astype(np.float32)
    client.put(save_path, image_embedding.tobytes())
    # image_embedding.tofile(save_path)

    


def parse_option():
    parser = argparse.ArgumentParser('SAM', add_help=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='/data/sets/nuScenes')
    parser.add_argument('-s', '--fea_folder', help='feature root', type=str,
                        default='s3://{save_root}/image_embedding/nuScenes/sam/') 
    parser.add_argument('-p', '--sam_checkpoint', help='path of pretrained model', type=str,
                        default='./sam_vit_h_4b8939.pth')
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_option()
    sam_checkpoint = args.sam_checkpoint
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)


    nuscenes_path = args.root_folder
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"

    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=False
    )
    # os.makedirs(args.fea_folder)

    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]
     # 850
    #  0， 26， 53，80， 107， 134， 161，188， 215，242，269，296，323，
    # 350，377，404，431，458，485， 512，539，566，593，620，647，674，701，728，755，782，809，836，850
    phase_scenes = create_splits_scenes()["train"]
    for scene_idx in tqdm(range(len(nusc.scene))):
        # print(scene_idx)
        # import pdb; pdb.set_trace()
        scene = nusc.scene[scene_idx]
        if scene["name"] in phase_scenes:
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                for camera_name in camera_list:
                    compute_sam(current_sample["data"][camera_name], args.fea_folder )

                current_sample_token = current_sample["next"]








