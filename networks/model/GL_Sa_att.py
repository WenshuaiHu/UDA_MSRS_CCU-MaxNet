# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:09:44 2024

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange
from typing import Optional

class Global_Local_Spatial_Attention(nn.Module):
                                 
    def __init__(self, dim, input_size, patch_size, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias=True, drop=0., attn_drop=0., dropout = 0.1, 
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(Global_Local_Spatial_Attention, self).__init__()

        # self.Swin = SwinTransformerBlocks(dim=dim, input_resolution=patch_size, depth=depth,
        #                                   num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
        #                                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                                   drop_path=drop_path, norm_layer=norm_layer)
        self.input_size = input_size
        self.num_heads = num_heads
        inner_dim = dim *  num_heads
        self.dim = dim
        self.inner_dim = inner_dim
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.scale = dim ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_outx = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.to_outy = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.Mtox =  nn.Linear(12, input_size**2)
        
        self.to_outM = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(nn.Linear(dim, inner_dim*3, bias = False))
            
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.to_q = nn.Linear(dim, inner_dim*3, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        
        
        
        
        
    def forward(self, x):

        x0 = x.permute(0, 3, 1, 2) ## B C H W
        x0 = x0.flatten(2).transpose(1, 2).contiguous()  ## 沿着第二个维度展成向量  B L D
        spatial_block = x0.chunk(len(self.qkvs), dim=1)  ## B L D//h h   h个分组
        
        feats_out = []
        for i, qkv in enumerate(self.qkvs):
            
            q, k, v = *qkv(spatial_block[i]).chunk(2, dim = -1)
            
            
            
            
            
            
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        