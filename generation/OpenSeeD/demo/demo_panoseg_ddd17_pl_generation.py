import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from utils.arguments import load_opt_command
from tqdm import tqdm
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer
import cv2
import random

logger = logging.getLogger(__name__)
VALIDATION_SET = {'dir1'}
TRAIN_SET = {'dir0','dir3','dir4','dir6',"dir7"}
# TEST_SET = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

def visual_2D_pred(image,pred):


    # 创建一个包含17个随机类别颜色的列表
    class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(11)]


    predicted = torch.argmax(pred, dim=1)
    predicted1 = predicted[0,:,:].detach().cpu().numpy()

    prediction_overlay = np.zeros_like(image, dtype=np.uint8)

    # 将每个类别的像素用不同颜色填充
    for class_id in range(len(class_colors)):
        if class_id == 30:
            class_mask = predicted1 == class_id
            class_color = class_colors[class_id]
            # import pdb; pdb.set_trace()
            prediction_overlay[class_mask] = (0,0,0)
        else:

            class_mask = predicted1 == class_id
            class_color = class_colors[class_id]
            # import pdb; pdb.set_trace()
            prediction_overlay[class_mask] = class_color

    # 将预测叠加到原始图像上
    alpha = 0.7  # 透明度
    output_image = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, prediction_overlay, alpha, 0)
    return output_image
    

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
    image_pth = './MASKCLIP/exp.png'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # thing_classes = ['car','person','traffic light', 'truck', 'motorcycle']
    # stuff_classes = ['building','sky','street','tree','rock','sidewalk']
    # CLASSES_DSEC = [
    #     'background',
    #     'building',
    #     'fence',
    #     'person',
    #     'pole',
    #     'road',
    #     'sidewalk',
    #     'vegetation', 
    #     'car', 
    #     'wall', 
    #     'traffic sign'

    # ]
    thing_classes = [ 'person','car']
    stuff_classes =[  'road' ,
                    'sky',
                    'building',
                    'pole',
                    'traffic sign',
                    'vegetation',
                    ]

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
    list_files = []
    # for num in {'zurich_city_00_a','zurich_city_01_a','zurich_city_02_a','zurich_city_04_a',
    #          'zurich_city_05_a','zurich_city_06_a','zurich_city_07_a',
    #          'zurich_city_08_a'}:
    VALIDATION_SET = {'dir1'}
    TRAIN_SET = {'dir0','dir3','dir4','dir6',"dir7"}
    for num in {'dir6'}:
        list_files += absoluteFilePaths('/'.join(['/data/sets/DDD17', num, 'images_aligned'])) #images_aligned/reconstructions
    for idx in tqdm(range(len(list_files))):
        print(idx)
        image_pth = list_files[idx]
        with torch.no_grad():
            image_ori = Image.open(image_pth).convert("RGB")
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)
            predicted = torch.argmax(outputs[-1]['sem_seg'].unsqueeze(0), dim=1)
            splits = image_pth.split('/')
            save_root = os.path.join('/ddd17_train_pl_19_unmerged',splits[-3],'images_aligned')
            save_path = os.path.join(save_root, splits[-1])
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            pl =  predicted[0,:,:].detach().cpu().numpy()

            
            mask_person = pl == 0
            mask_car = pl == 1
            mask_road = pl == 2
            mask_sky = pl ==3
            mask_build = pl == 4
            mask_pole = pl ==5
            mask_traff = pl ==6
            mask_veg = pl ==7


            pl[mask_road] = 0
            pl[mask_sky] = 1
            pl[mask_build] = 2
            pl[mask_pole] = 3
            pl[mask_traff] = 4
            pl[mask_veg] = 5
            pl[mask_person] = 6
            pl[mask_car] = 7



            im = Image.fromarray(pl.astype(np.uint8))
            # import pdb; pdb.set_trace()
            im.save(save_path)



if __name__ == "__main__":
    main()
    sys.exit(0)