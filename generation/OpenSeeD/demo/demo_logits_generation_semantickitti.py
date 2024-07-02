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
logger = logging.getLogger(__name__)

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def select_points_in_frustum(points_2d, x1, y1, x2, y2):
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

def read_calib(calib_path):
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

    thing_classes = [ "car", "bicycle", "motorcycle", "truck", "other vehicle", "person", "bicyclist", "motorcyclist"]
    stuff_classes =[ "road", "parking", "sidewalk", "other ground", "building", "fence", "tree", "trunk", "terrain", "pole", "traffic sign","sky"]

    # thing_classes = [ "barrier", 
    #               "bicycle", 
    #               "bus" , 
    #               "car",
    #              "construction vehicle", 
    #              "motorcycle", 
    #              "person", 
    #              "traffic cone", 
    #              "trailer", 
    #              "truck"]
    # stuff_classes =[ "road", 
    #              "other flat", 
    #              "sidewalk", 
    #              "terrain", 
    #              "manmade", 
    #              "tree",
    #              "sky"]

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



    seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    root_path = "/data/sets/semantickitti/sequences/"
    annos = []
    for seq in seqs:
       annos += absoluteFilePaths('/'.join([root_path, str(seq).zfill(2), 'velodyne']))

    sp_root = 's3://{save_root}/image_logits/semantickitti/openseed/'

    # [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800, 8400,
    #  9000, 9600, 10200, 10800, 11400, 12000, 12600, 13200, 13800, 14400, 15000, 15600, 16200, 16800, 17400, 18000, 18600, 19200]
    with torch.no_grad():
        for idx in tqdm(range(len(annos))):
            print(idx)
            path = annos[idx]
            points = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
            path_splits = path.split('/')
            calib_path = os.path.join("/data/sets/semantickitti/sequences",path_splits[-3], "calib.txt")
            
            image_ori = cv2.imread(os.path.join("/data/sets/semantickitti/images/sequences/", path_splits[-3],"image_2", path_splits[-1].replace("bin", "png")))
            image_ori = cv2.resize(image_ori, (1241, 376), interpolation=cv2.INTER_LINEAR)
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
            image_ori = Image.fromarray(image_ori)
            width = image_ori.size[0]
            height = image_ori.size[1]


            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()


            calib = read_calib(calib_path)
            proj_matrix = calib['P2'] @ calib['Tr']
            proj_matrix = proj_matrix.astype(np.float32)

            # project points into image
            keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
            points_hcoords = np.concatenate([points[:, :3], np.ones([len(points), 1], dtype=np.float32)], axis=1)
            img_points = (proj_matrix @ points_hcoords.T).T
            matching_pixel = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
            

            # print(img_points)
            keep_idx_img_pts = select_points_in_frustum(matching_pixel, 0, 0, 1241, 376)
            # print(keep_idx)
            keep_idx = keep_idx_img_pts & keep_idx
            matching_pixel = matching_pixel[keep_idx]
            matching_pixel = np.fliplr(matching_pixel).astype(np.int64)

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)

            image_logits = outputs[-1]['sem_seg'][:-1,:,:].permute(1,2,0).contiguous().detach().cpu().numpy()
            image_logits = image_logits[matching_pixel[:, 0], matching_pixel[:, 1],:]

            save_path = sp_root + "/sequences/"+ path_splits[-3] +'/' + path_splits[-1]
            image_logits = image_logits.astype(np.float32)
            client.put(save_path, image_logits.tobytes())
            # visual_2D_pred(image_ori,outputs[-1]['sem_seg'].unsqueeze(0))
        



if __name__ == "__main__":
    main()
    sys.exit(0)