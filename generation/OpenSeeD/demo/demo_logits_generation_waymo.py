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
import pickle
import io
import copy
logger = logging.getLogger(__name__)


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


    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)


    thing_classes = [ "car",  "truck", "bus","other vehicle", "motorcyclist","bicyclist","person", "bicycle", "motorcycle",]
    stuff_classes =[ "traffic sign","traffic light","pole","construction cone", "building",
                    "vegetation", "tree trunk", "curb", "road", "lane marker", "other ground", "walkable", "sidewalk", "sky"]
    


    # pl= None

    # mask_sign = pl == 9
    # mask_light = pl == 10
    # mask_pole = pl == 11
    # mask_cone  = pl == 12
    # mask_bicycle = pl == 7
    # mask_motor = pl == 8

    # pl[mask_sign] = 7
    # pl[mask_light] = 8
    # pl[mask_pole] = 9
    # pl[mask_cone] = 10
    # pl[mask_bicycle] = 11
    # pl[mask_motor] = 12

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

    calib_info = '/data/sets/waymo/images_infos_new.pkl'
    df = open(calib_info,'rb')
    data = pickle.load(df)
    annos = []
    with open('/data/sets/waymo/train-0-31.txt', 'r') as f:
        for line in f.readlines():
                annos.append(line.strip())

    sp_root = 's3://{save_root}/image_logits/waymo/openseed/'
    image_root = "/data/sets/waymo/images/"

    # [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200, 7800, 8400,
    #  9000, 9600, 10200, 10800, 11400, 12000, 12600, 13200, 13800, 14400, 15000, 15600, 16200, 16800, 17400, 18000, 18600, 19200]
    with torch.no_grad():
        for idx in tqdm(range(len(annos))[21000:]): #16800:
            print(idx)
            ann_info = annos[idx]
            
            raw_xyz = np.load(io.BytesIO(client.get(ann_info))).reshape(-1,7)[:,3:6].reshape((-1,3)).astype(np.float32)
            intenel = np.load(io.BytesIO(client.get(ann_info))).reshape(-1,7)[:,1:3].reshape((-1,2)).astype(np.float32)
            pc_first = np.concatenate((raw_xyz,intenel),1)

            sec_path = ann_info.replace('first/', 'second/')
            raw_xyz1 = np.load(io.BytesIO(client.get(sec_path))).reshape(-1,7)[:, 3:6].reshape((-1, 3)).astype(np.float32)
            intenel1 = np.load(io.BytesIO(client.get(sec_path))).reshape(-1,7)[:, 1:3].reshape((-1, 2)).astype(np.float32)
            pc_second = np.concatenate((raw_xyz1, intenel1), 1)
            points_ = np.concatenate((pc_first, pc_second), 0).astype(np.float32).copy()
            image_infos = data[ann_info.split('/')[-1]]
            image_info = image_infos['image']
            sample_idx = image_infos['sample_idx']
            sequence_name = image_infos['sequence_name']
            for i in range(5):
                points = copy.deepcopy(points_)
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
                image_ori = cv2.imread(image_path_i)
                image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

                image_ori = Image.fromarray(image_ori)
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
                proj_matrix = image_intrinsic_i @ image_extrinsic_i
                proj_matrix = proj_matrix.astype(np.float32)
                        # project points into image
                points_hcoords = np.concatenate([points[:, :3], np.ones([len(points), 1], dtype=np.float32)], axis=1)
                img_points = (proj_matrix @ points_hcoords.T).T
                matching_pixel = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
                keep_idx_img_pts = select_points_in_frustum(matching_pixel, 0, 0, image_shape_i[1], image_shape_i[0]) & (img_points[:, 2] > 0)
                # print(keep_idx)
                keep_idx = keep_idx_img_pts 
                matching_pixel = matching_pixel[keep_idx]
                matching_pixel = np.fliplr(matching_pixel).astype(np.int64)
                batch_inputs = [{'image': images, 'height': height, 'width': width}]
                outputs = model.forward(batch_inputs)
                image_logits = outputs[-1]['sem_seg'][:-1,:,:].permute(1,2,0).contiguous().detach().cpu().numpy()
                # import pdb;pdb.set_trace()
                mask_sign = image_logits[:,:,9]
                mask_light = image_logits[:,:,10] 
                mask_pole = image_logits[:,:,11]
                mask_cone  = image_logits[:,:,12] 
                mask_bicycle = image_logits[:,:,7] 
                mask_motor = image_logits[:,:,8] 

                   

                image_logits[:,:,7] = mask_sign
                image_logits[:,:,8] = mask_light
                image_logits[:,:,9] = mask_pole
                image_logits[:,:,10] = mask_cone
                image_logits[:,:,11] = mask_bicycle
                image_logits[:,:,12] = mask_motor


                image_logits = image_logits[matching_pixel[:, 0], matching_pixel[:, 1],:]
                assert image_logits.shape[1] == 22
                 
                # image_root +sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id

                save_path = sp_root + sequence_name.replace('_with_camera_labels','') + '/' + 'image_'+str(i) +'/'+image_id.replace('jpg','bin')
                # print(save_path)
                image_logits = image_logits.astype(np.float32)
                client.put(save_path, image_logits.tobytes())
            # visual_2D_pred(image_ori,outputs[-1]['sem_seg'].unsqueeze(0))
        



if __name__ == "__main__":
    main()
    sys.exit(0)