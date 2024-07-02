import torch
from torch import nn
from torch.nn import functional as F
# from _resnet import resnet50
from . import _resnet as resnet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from ..builder import MODELS
import io
class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# def segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
#     if output_stride == 8:
#         replace_stride_with_dilation = [False, True, True]
#         aspp_dilate = [12, 24, 36]
#     else:
#         replace_stride_with_dilation = [False, False, True]
#         aspp_dilate = [6, 12, 18]
#
#     backbone = resnet.__dict__[backbone_name](
#         pretrained=pretrained_backbone,
#         replace_stride_with_dilation=replace_stride_with_dilation)
#
#     inplanes = 2048
#     low_level_planes = 256
#
#     if name == 'deeplabv3plus':
#         return_layers = {'layer4': 'out', 'layer1': 'low_level'}
#         classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
#     elif name == 'deeplabv3':
#         return_layers = {'layer4': 'out'}
#         classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
#     backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
#
#     model = DeepLabV3(backbone, classifier)
#     return model


class DeepLabHead(nn.Module):
    def __init__(self, config, in_channels,aspp_dilate=[12, 24, 36], norm_fn=None):
        super(DeepLabHead, self).__init__()

        self.ASPP = ASPP(in_channels, aspp_dilate, norm_fn=norm_fn)

        self.pixel_feature = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        
        # classifier
        
        # self.conv1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        # self.bn1 =  norm_fn(512)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conditions =("SemanticKITTI", "nuScenes") 
        num_classes = (19, 16)
        self.seg_heads = nn.ModuleList([
            nn.Conv2d(256, num_cls,1) for num_cls in num_classes
        ])
            # nn.Conv2d(256, num_classes, 1)
        

        self._init_weight()
        self.config = config
        # TODO condition head
        # self.text_embeddings_nu_path = self.config['text_embeddings_nu_path']
        # text_categories_nu = self.config['text_categories_nu']
        # if self.text_embeddings_nu_path is None:
        #     self.text_embeddings_nu= nn.Parameter(torch.zeros(text_categories_nu, 512))
        #     nn.init.normal_(self.text_embeddings_nu, mean=0.0, std=0.01)
        # else:
        #     with open(self.text_embeddings_nu_path, 'rb') as f:
        #         buffer = io.BytesIO(f.read())

        #     # Load data from the buffer
        #     buffer.seek(0)
        #     loaded_data = torch.load(buffer, map_location='cuda')

        #     # Create a parameter with the loaded data
        #     loaded_parameter = nn.Parameter(loaded_data)

        #     # Register the loaded parameter as a buffer
        #     self.register_buffer('text_embeddings_nu', loaded_parameter)


            # self.register_buffer('text_embeddings_nu', torch.randn(text_categories_nu, 512))
            # loaded = torch.load(self.text_embeddings_nu, map_location='cpu')
            # self.text_embeddings_nu[:, :] = loaded[:, :]


        # self.text_embeddings_sk_path = self.config['text_embeddings_sk_path']
        # text_categories_sk = self.config['text_categories_sk']
        # if self.text_embeddings_sk_path is None:
        #     self.text_embeddings_sk= nn.Parameter(torch.zeros(text_categories_sk, 512))
        #     nn.init.normal_(self.text_embeddings_sk, mean=0.0, std=0.01)
        # else:
        #     with open(self.text_embeddings_sk_path, 'rb') as f:
        #         buffer = io.BytesIO(f.read())

        #     # Load data from the buffer
        #     buffer.seek(0)
        #     loaded_data = torch.load(buffer, map_location='cuda')

        #     # Create a parameter with the loaded data
        #     loaded_parameter = nn.Parameter(loaded_data)

        #     # Register the loaded parameter as a buffer
        #     self.register_buffer('text_embeddings_sk', loaded_parameter)

            # self.register_buffer('text_embeddings_sk', torch.randn(text_categories_sk, 512))
            # loaded = torch.load(self.text_embeddings_sk, map_location='cpu')
            # self.text_embeddings_sk[:, :] = loaded[:, :]
        # self.text_embeddings = torch.cat((self.text_embeddings[0, :].unsqueeze(0)*0, self.text_embeddings), dim=0)

    def forward(self, feature, condition):
        feature = self.ASPP([feature,condition])

        # feature = self.conv1(feature)
        # feature = self.bn1(feature,condition)
        # feature = self.relu1(feature)

        # feats = self.pixel_feature(feature)
        # print(self.text_embeddings_nu[:, :, None, None].shape)
        # print(self.text_embeddings_sk[:, :, None, None].shape)
        
        # prediction = F.conv1d(global_feature.F.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()
        # if condition == "nuScenes":
        #     logist = F.conv2d(feature, self.text_embeddings_nu[:, :, None, None])
        # elif condition == "SemanticKITTI":
        #     logist = F.conv2d(feature, self.text_embeddings_sk[:, :, None, None])
        # print(logist.shape)
        # import pdb; pdb.set_trace()
        seg_head = self.seg_heads[self.conditions.index(condition)]
        logist = seg_head(feature)
    

        return logist,feature

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PDBatchNorm(torch.nn.Module):
    def __init__(self,
                 num_features,
                 conditions=("ScanNet", "S3DIS", "Structured3D"),
                 decouple=True,
                 ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        if self.decouple:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(num_features=num_features)
                for _ in conditions
            ])
        else:
            self.bn = nn.BatchNorm2d(num_features=num_features)


    def forward(self, feat, condition=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        return feat

@MODELS.register_module("DeepLabV3")
class deeplabv3_resnet50(nn.Module):
    def __init__(self, config):
        super(deeplabv3_resnet50, self).__init__()
        self.config = config

        # num_classes = config["n_classes"]
        output_stride = config["output_stride"]
        self.conditions = config["conditions"]
        pretrained_backbone = config["pre_trained_backbone"]
        norm_fn = partial(PDBatchNorm,conditions=self.conditions)

        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]

        backbone = resnet.__dict__['resnet50'](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation,norm_layer=norm_fn)
        # backbone = resnet50(
        #     pretrained=pretrained_backbone,
        #     replace_stride_with_dilation=replace_stride_with_dilation,norm_layer=norm_fn)


        inplanes = 2048
        classifier = DeepLabHead(config, inplanes, aspp_dilate, norm_fn=norm_fn)
        # backbone = IntermediateLayerGetter(backbone, return_layers={'layer4': 'out'})

        self.backbone = backbone
        self.classifier = classifier

    def forward(self, input_dict):
        x = input_dict["images"]
        condition = input_dict["condition"][0]
        input_shape = x.shape[-2:]
        features,_ = self.backbone([x,condition])
        # import pdb;pdb.set_trace()
        logist, feats = self.classifier(features, condition)
        logist = F.interpolate(logist, size=input_shape, mode='bilinear', align_corners=False)
        # feats = F.interpolate(feats, size=input_shape, mode='bilinear', align_corners=False)
        return logist, feats


# def load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
#     model = segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
#                         pretrained_backbone=pretrained_backbone)
#     return model


#
# def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone='imagenet'):
#     """Constructs a DeepLabV3 model with a ResNet-50 backbone.
#
#     Args:
#         num_classes (int): number of classes.
#         output_stride (int): output stride for deeplab.
#         pretrained_backbone (bool): If True, use the pretrained backbone.
#     """
#     return load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride,
#                       pretrained_backbone=pretrained_backbone)



class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_fn=None):
        super(ASPPConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_fn(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x, condition = x
        out = self.conv1(x)
        out = self.bn1(out, condition)
        out = self.relu1(out)
        return out


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn=None):
        super(ASPPPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 =  norm_fn(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x, condition = x
        size = x.shape[-2:]
        out = self.pool1(x)
        out = self.conv1(out)
        out = self.bn1(out, condition)
        out = self.relu1(out)

        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates,norm_fn):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
   
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = norm_fn(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.asp1 = ASPPConv(in_channels, out_channels, rate1, norm_fn=norm_fn)
        self.asp2= ASPPConv(in_channels, out_channels, rate2, norm_fn=norm_fn)
        self.asp3= ASPPConv(in_channels, out_channels, rate3 , norm_fn=norm_fn)
        self.asp4= ASPPPooling(in_channels, out_channels, norm_fn=norm_fn)

        # self.convs = nn.ModuleList(modules)

        
        self.conv2 = nn.Conv2d(5 * out_channels, out_channels, 1, bias=False)
        self.bn2 = norm_fn(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x_):
        x, condition = x_
        res = []
        # import pdb; pdb.set_trace()
        out1 = self.conv1(x)
        out1 = self.bn1(out1, condition)
        out1 = self.relu1(out1)
        # print(out1.shape)
        # import pdb; pdb.set_trace()
        res.append(out1)

        out2 = self.asp1([x,condition])
        # print(out2.shape)
        res.append(out2)
        out3 = self.asp2([x,condition])
        # print(out3.shape)
        res.append(out3)
        out4 = self.asp3([x, condition])
        # print(out4.shape)
        res.append(out4)
        out5 = self.asp4([x, condition])
        # print(out5.shape)
        res.append(out5)

        res = torch.cat(res, dim=1)

        out = self.conv2(res)
        out = self.bn2(out, condition)
        out = self.relu2(out)
        out = self.drop(out)

        return out

# def visual_2D_pred(image,pred):


#    
#     class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(17)]

#     image1 = image[0,:,:,:].cpu().numpy().transpose(1, 2, 0) * 255.0
#     predicted = torch.argmax(pred, dim=1)
#     predicted1 = predicted[0,:,:].detach().cpu().numpy()

#     prediction_overlay = np.zeros_like(image1, dtype=np.uint8)

#     
#     for class_id in range(len(class_colors)):
#         class_mask = predicted1 == class_id
#         class_color = class_colors[class_id]
#         # import pdb; pdb.set_trace()
#         prediction_overlay[class_mask] = class_color

#     
#     alpha = 0.7  
#     output_image = cv2.addWeighted(image1.astype(np.uint8), 1 - alpha, prediction_overlay, alpha, 0)

#     cv2.imwrite("./vis_exp" +  '.png', output_image)

# if __name__ == "__main__":
#     import cv2
#     from PIL import Image
#     import random
#     configs_ = {'text_embeddings_path':'/mnt/lustre/konglingdong/youquan/CNS/utils/nuscenes_ViT16_clip_text_i.pth', 
#                       "n_classes":16, "text_categories":16,
#         "output_stride": 32,
#         "conditions":["nuScenes","SemanticKITTI","Waymo"],
#         "pre_trained_backbone":"imagenet"}
#     image_path = "/mnt/lustre/konglingdong/youquan/MASKCLIP/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297148912404.jpg"
#     images = []
#     imageDim = (416, 224)
#     im = np.array(Image.open(image_path))
#     # im = images.append(cv2.resize(im, imageDim) / 255)
#     images.append(im / 255)
#     im = images.append(im / 255)
#     images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2)).cuda()
#     model = deeplabv3_resnet50(config=configs_)
#     model = model.cuda()
#     pred,feat = model([images,'nuScenes'])
#     visual_2D_pred(images,pred)
    