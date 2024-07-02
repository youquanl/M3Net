import numpy as np
import torch
import matplotlib.pyplot as plt
from petrel_client.client import Client
client = Client('~/.petreloss.conf')
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import io
import copy




def parse_option():
    parser = argparse.ArgumentParser('SAM', add_help=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default="/data/sets/waymo/")
    parser.add_argument('-s', '--fea_folder', help='feature root', type=str,
                        default='s3://{save_root}/image_embedding/waymo/sam/') 
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
    sp_root = args.fea_folder


    calib_info = '/data/sets/waymo/images_infos_new.pkl'
    image_root = "/data/sets/waymo/images/"
    df = open(calib_info,'rb')
    data = pickle.load(df)
    annos = []
    with open('/data/sets/waymo/train-0-31.txt', 'r') as f:
        for line in f.readlines():
                annos.append(line.strip())

    for idx in tqdm(range(len(annos))):
        # print(idx)
        ann_info = annos[idx]
        image_infos = data[ann_info.split('/')[-1]]
        image_info = image_infos['image']
        sample_idx = image_infos['sample_idx']
        sequence_name = image_infos['sequence_name']
        for i in range(5):
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
            image_path_i =  image_root +sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id
            image = cv2.imread(image_path_i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding().detach().cpu().numpy()

            save_path = sp_root + sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id.replace('png','bin')
            image_embedding = image_embedding.astype(np.float32)
            client.put(save_path, image_embedding.tobytes())










