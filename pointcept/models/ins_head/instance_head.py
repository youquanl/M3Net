# Reference: https://github.com/hongfz16/DS-Net
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from functools import partial
from collections import OrderedDict
from ..builder import MODELS


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)

@MODELS.register_module("Ins-Head")
class ins_head(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.pt_fea_dim = in_channels
        init_size = 32

        self.conv1 = conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='offset_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.ReLU()
        self.conv2 = conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='offset_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.ReLU()
        self.conv3 = conv3x3(2 * init_size, init_size, indice_key='offset_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.ReLU()

        self.offset = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(init_size, 3, bias=True)

    def forward(self, x, batch):
        x = self.conv1(x)
        x = x.replace_feature(self.bn1(x.features))
        x = x.replace_feature(self.act1(x.features))

        x = self.conv2(x)
        x = x.replace_feature(self.bn2(x.features))
        x = x.replace_feature(self.act2(x.features))

        x = self.conv3(x)
        x = x.replace_feature(self.bn3(x.features))
        x = x.replace_feature(self.act3(x.features))
  

        pt_ins_fea_list = []
        invs = batch['inverse_indexes']
        for idx in range(len(batch["batch_index"])-1):
            cur_inv = invs[batch["batch_index"][idx]:batch["batch_index"][idx+1]] 
            outputs_mapped = x.features[cur_inv]
            pt_ins_fea_list.append(outputs_mapped)

        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(
                self.offset_linear(
                self.offset(torch.cat([pt_ins_fea, batch['org_coord'][batch["batch_index"][batch_i]:batch["batch_index"][batch_i+1]].cuda()], dim=1))))
        
        
        return pt_pred_offsets_list