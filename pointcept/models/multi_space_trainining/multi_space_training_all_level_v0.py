from functools import partial
from collections import OrderedDict
from torch.nn import functional as F
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
import io

def calculate_offset_loss(pt_offsets, gt_offsets, valid):
    offset_diff = pt_offsets - gt_offsets  # (N, 3)
    offset_dist = torch.sum(torch.abs(offset_diff), dim=-1)  # (N)
    valid_mask = valid.view(-1).float()
    normalized_loss = torch.sum(offset_dist * valid_mask) / (torch.sum(valid_mask) + 1e-6)
    return (normalized_loss,)

def compile_offset_loss(single_offset_loss_fn):
    def compute_offset_loss(pt_offsets_list, gt_offsets_list, valid_list, gt_semantic_label=None, xyz=None):
        loss_lists = []
        for i in range(len(pt_offsets_list)):
            individual_loss_list = single_offset_loss_fn(pt_offsets_list[i], gt_offsets_list[i], valid_list[i])
            num_losses = len(individual_loss_list)
            if len(loss_lists) < num_losses:
                loss_lists = [[] for _ in range(num_losses)]
            for j in range(num_losses):
                loss_lists[j].append(individual_loss_list[j])
        averaged_loss_list = []
        for i in range(len(loss_lists)):
            averaged_loss_list.append(torch.mean(torch.stack(loss_lists[i])))
        return averaged_loss_list

    return compute_offset_loss

offset_loss_calculator = compile_offset_loss(calculate_offset_loss)

@MODELS.register_module("all-level-v0")
class MultiSpaceTraining(nn.Module):
    def __init__(self,
                 backbone=None,
                 ins_head=None,
                 domain_attention=None,
                 criteria=None,
                 backbone_out_channels=96,
                 context_channels=256,
                 conditions=("Structured3D", "ScanNet", "S3DIS"),
                 num_classes=(25, 20, 13)
                 ):
        super().__init__()
        assert len(conditions) == len(num_classes)
        self.backbone = MODELS.build(backbone)


        self.ins_head_sk = MODELS.build(ins_head)
        self.ins_head_nu = MODELS.build(ins_head)
        self.feat256 = nn.Linear(backbone_out_channels, 256)

        self.criteria = build_criteria(criteria)
        self.ins_loss = offset_loss_calculator

        self.conditions = conditions

        
        self.proj_head = nn.Linear(backbone_out_channels, 512)

        # nusc
        self.text_embeddings_nu_path = './text_prompt/nuscenes_ViT16_clip_text_p.pth'  # prompt engineering from CLIP
        with open(self.text_embeddings_nu_path, 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load data from the buffer
        buffer.seek(0)
        loaded_data = torch.load(buffer, map_location='cpu')

        # Create a parameter with the loaded data
        loaded_parameter = nn.Parameter(loaded_data)

        # Register the loaded parameter as a buffer
        self.register_buffer('text_embeddings_nu', loaded_parameter)

        # kitti
        self.text_embeddings_sk_path = './text_prompt/kitti_ViT16_clip_text_p.pth'

        with open(self.text_embeddings_sk_path, 'rb') as f:
            buffer_sk = io.BytesIO(f.read())

        # Load data from the buffer
        buffer_sk.seek(0)
        loaded_data_sk = torch.load(buffer_sk, map_location='cpu')

        # Create a parameter with the loaded data
        loaded_parameter_sk = nn.Parameter(loaded_data_sk)

        # Register the loaded parameter as a buffer
        self.register_buffer('text_embeddings_sk', loaded_parameter_sk)

        # waymo
        self.text_embeddings_wa_path = './text_prompt/waymo_ViT16_clip_text_p.pth'
        with open(self.text_embeddings_wa_path, 'rb') as f:
            buffer_wa = io.BytesIO(f.read())

        # Load data from the buffer
        buffer_wa.seek(0)
        loaded_data_wa = torch.load(buffer_wa, map_location='cpu')

        # Create a parameter with the loaded data
        loaded_parameter_wa = nn.Parameter(loaded_data_wa)

        # Register the loaded parameter as a buffer
        self.register_buffer('text_embeddings_wa', loaded_parameter_wa)

    def forward(self, data_dict):
        condition = data_dict["condition"][0]
        assert condition in self.conditions
        feat,sp_feat = self.backbone(data_dict)
        feat_256c=self.feat256(feat)
        feat = self.proj_head(feat)
        pred_offsets = []
        # import pdb; pdb.set_trace()
        if condition == 'nuScenes':
            seg_logits = F.conv1d(feat.unsqueeze(-1), self.text_embeddings_nu[:, :, None].cuda()).squeeze()
            pred_offsets = self.ins_head_nu(sp_feat, data_dict)
        elif condition == 'SemanticKITTI':
            seg_logits = F.conv1d(feat.unsqueeze(-1), self.text_embeddings_sk[:, :, None].cuda()).squeeze()
            pred_offsets = self.ins_head_sk(sp_feat, data_dict)
        elif condition == 'Waymo':
            seg_logits = F.conv1d(feat.unsqueeze(-1), self.text_embeddings_wa[:, :, None].cuda()).squeeze()

 
        loss_ins = 0
        loss_logits =0
        loss_fea = 0
              
        if self.training:
            pt_valid = []
            pt_offsets = []
            if condition == "SemanticKITTI" or condition == "nuScenes":
                for idx in range(len(data_dict["batch_index"])-1):
                    pt_valid.append(data_dict['valid'][data_dict["batch_index"][idx]:data_dict["batch_index"][idx+1]])
                    pt_offsets.append(data_dict['gt_off'][data_dict["batch_index"][idx]:data_dict["batch_index"][idx+1]])
                
                offset_loss_list = self.ins_loss(pred_offsets, pt_offsets, pt_valid)
                loss_ins = sum(offset_loss_list)
                pairing_points = data_dict["pairing_points"]
                sam_features = data_dict["sam_features"]
                loss_fea = torch.mean(1 - F.cosine_similarity(sam_features, feat_256c[pairing_points], dim=1))

        
                img_logits = data_dict["img_logits"]
                img_logits = F.normalize(img_logits, p=2, dim=1)
                p_logits = F.normalize(seg_logits[pairing_points], p=2, dim=1)

                loss_logits = torch.mean(1 - F.cosine_similarity(img_logits, p_logits, dim=1))
            
            loss = self.criteria(seg_logits, data_dict["segment"]) + loss_fea +   loss_ins  + loss_logits
            return dict(loss=loss)

        # eval
        elif "segment" in data_dict.keys():

            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits, pred_offsets=pred_offsets)