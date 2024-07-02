from functools import partial
from collections import OrderedDict
from torch.nn import functional as F
import torch
import torch.nn as nn
import numpy as np
from pointcept.models.builder import MODELS

class AttentionLayer(nn.Module):
    def __init__(self, channels, reduction_ratio=16, apply_sigmoid=True):
        super(AttentionLayer, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        if apply_sigmoid:
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction_ratio, channels),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction_ratio, channels),
            )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return y

@MODELS.register_module("DomainSpecificAttention")
class DomainSpecificAttention(nn.Module):
    def __init__(self, channels=256, reduction_ratio=16, class_counts=None, use_fixed_block=False):
        super(DomainSpecificAttention, self).__init__()
        self.channels = channels
        self.use_fixed_block = use_fixed_block
        adapters_count = 3
        self.num_domains = adapters_count if adapters_count != 0 else len(class_counts)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if self.use_fixed_block or adapters_count == 1:
            self.attention_layers = nn.ModuleList([AttentionLayer(channels, reduction_ratio, apply_sigmoid=False) for _ in range(1)])
        elif adapters_count == 0:
            self.attention_layers = nn.ModuleList([AttentionLayer(channels, reduction_ratio, apply_sigmoid=False) for _ in class_counts])
        else:
            self.attention_layers = nn.ModuleList([AttentionLayer(channels, reduction_ratio, apply_sigmoid=False) for _ in range(adapters_count)])
        
        self.fc1 = nn.Linear(channels, self.num_domains)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channels, _, _ = x.size()

        if self.use_fixed_block:
            attention_matrix = self.attention_layers[0](x).view(batch, channels, 1, 1)
            attention_matrix = self.sigmoid(attention_matrix)
        else:
            domain_weights = self.fc1(self.global_avg_pool(x).view(batch, channels))
            domain_weights = self.softmax(domain_weights).view(batch, self.num_domains, 1)
            for idx, attention_layer in enumerate(self.attention_layers):
                if idx == 0:
                    attention_matrix = attention_layer(x).view(batch, channels, 1)
                else:
                    attention_matrix = torch.cat((attention_matrix, attention_layer(x).view(batch, channels, 1)), 2)
            attention_matrix = torch.matmul(attention_matrix, domain_weights).view(batch, channels, 1, 1)
            attention_matrix = self.sigmoid(attention_matrix)
        
        return x * attention_matrix + x

