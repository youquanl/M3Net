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


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def compute_sam(path, sp_root):
    path_splits = path.split('/')
    image = cv2.imread(os.path.join("/data/sets/semantickitti/imgages/sequences/", path_splits[-3],"image_2", path_splits[-1].replace("bin", "png")))
    image = cv2.resize(image, (1241, 376), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # masks = mask_generator.generate(image)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().detach().cpu().numpy()

    save_path = sp_root + "/sequences/"+ path_splits[-3] +'/' + path_splits[-1]
    image_embedding = image_embedding.astype(np.float32)
    client.put(save_path, image_embedding.tobytes())

    


def parse_option():
    parser = argparse.ArgumentParser('SAM', add_help=False)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default="/data/sets/semantickitti/sequences")
    parser.add_argument('-s', '--fea_folder', help='feature root', type=str,
                        default='s3://{save_root}/image_embedding/semantickitti/sam/') 
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


    seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    root_path = args.root_folder
    annos = []
    for seq in seqs:
       annos += absoluteFilePaths('/'.join([root_path, str(seq).zfill(2), 'velodyne']))
    # [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800, 8400,
    #  9000, 9600, 10200, 10800, 11400, 12000, 12600, 13200, 13800, 14400, 15000, 15600, 16200, 16800, 17400, 18000, 18600, 19200]

    for idx in tqdm(range(len(annos))):
        # print(idx)
        path = annos[idx]
        # import pdb; pdb.set_trace()
        compute_sam(path, args.fea_folder )










