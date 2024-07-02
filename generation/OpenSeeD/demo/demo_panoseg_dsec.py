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

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer
import cv2
import random

logger = logging.getLogger(__name__)

def visual_2D_pred(image,pred):


    # 创建一个包含17个随机类别颜色的列表
    class_colors = {
    0: (0, 150, 255),    # background
    1: (118, 118, 118),  # building
    2: (214, 220, 229),  # fence
    3: (4, 50, 255),     # person
    4: (190, 153, 153),  # pole
    5: (155, 55, 255),   # road
    6: (102, 102, 156),  # sidewalk
    7: (0, 176, 80),     # vegetation
    8: (250, 188, 1),    # car
    9: (152, 251, 152),  # wall
    10: (255, 0, 0),     # traffic-sign
}
    # class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(11)]


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

    cv2.imwrite("./vis_dsec1" +  '.png', output_image)

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
    image_pth = './exp.png'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # thing_classes = ['car','person','traffic light', 'truck', 'motorcycle']
    # stuff_classes = ['building','sky','street','tree','rock','sidewalk']

    thing_classes = [ 'person','car',]
    stuff_classes =[   'background','building', 'fence','pole',
    'road',
    'sidewalk',
    'vegetation', 
    'wall', 
    'traffic sign']

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

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        # print(image.shape)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        # print(outputs[-1]['sem_seg'].unsqueeze(0)[:,:-1,:,:].shape)
        visual_2D_pred(image_ori,outputs[-1]['sem_seg'].unsqueeze(0))



if __name__ == "__main__":
    main()
    sys.exit(0)