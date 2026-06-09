# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:36:04 2024

@author: user
"""

import torch
from torch import nn, einsum
import torch.nn.functional as func
import torch.nn.functional as F
from einops import rearrange
from ..model.MDLSTM import MDLSTM_down, MDLSTM_up, SwinTransformerBlocks
from ..model.GL_Se_att import Mlp
class Dynamic_fusion_Block1(torch.nn.Module):
    def __init__(self, dim=32, out_dim = 64, ffn=True, drop_path=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, 1, 1)     
        self.avg1  = nn.AvgPool1d((1))
        
        self.fc1 = nn.Linear(dim, out_dim)
        self.fc2 = nn.Linear(dim, out_dim)
        self.conv0 = nn.Conv2d(dim, dim, 1, 1)  
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        
        self.ffn = ffn
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.ffn:
            self.norm = norm_layer(out_dim*2)
            mlp_hidden_dim = int(out_dim*2 * mlp_ratio)
            self.mlp = Mlp(
                in_features=out_dim*2,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        
        
    def forward(self, x, M_curr, y):

        #M_curr = self.avg1(M_curr)
        M_curr = torch.mean(M_curr, dim=1)

        M_curr_up = self.softmax1(self.fc1(M_curr)).unsqueeze(dim=2).unsqueeze(dim=3)
        M_curr_iden = self.softmax2(self.fc2(M_curr)).unsqueeze(dim=2).unsqueeze(dim=3)
        
        f1 = M_curr_up * x + x
        f2 = M_curr_iden * y + y
        
        #f = f1 + f2
        f = torch.cat((f1, f2), 1)


        if self.ffn:
            f = rearrange(f, 'B C H W-> B H W C')##4 32 32 128
            f = f + self.drop_path(self.mlp(self.norm(f)))
            f = rearrange(f, 'B H W C -> B C H W')##4 128 32 32
            
        return f
    
    
class Dynamic_fusion_Block2(torch.nn.Module):
    def __init__(self, dim=32, out_dim = 64, ffn=True, drop_path=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, 1, 1)     
        self.avg1  = nn.AvgPool1d((1))
        
        self.fc1 = nn.Linear(dim, out_dim)
        self.fc2 = nn.Linear(dim, out_dim//2)
        self.conv0 = nn.Conv2d(dim, dim, 1, 1)  
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        
        self.ffn = ffn
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.ffn:
            self.norm = norm_layer(out_dim//2*3)
            mlp_hidden_dim = int(out_dim//2*3 * mlp_ratio)
            self.mlp = Mlp(
                in_features=out_dim//2*3,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, M_curr, y):

        #M_curr = self.avg1(M_curr)
        M_curr = torch.mean(M_curr, dim=1)
        
        M_curr_up = self.softmax1(self.fc1(M_curr)).unsqueeze(dim=2).unsqueeze(dim=3)
        M_curr_iden = self.softmax2(self.fc2(M_curr)).unsqueeze(dim=2).unsqueeze(dim=3)

        f1 = M_curr_up * x + x
        f2 = M_curr_iden * y + y
        
        #f = f1 + f2
        
        f = torch.cat((f1, f2), 1)
    
        if self.ffn:
            f = rearrange(f, 'B C H W-> B H W C')##4 32 32 192
            f = f + self.drop_path(self.mlp(self.norm(f)))
            f = rearrange(f, 'B H W C -> B C H W')##4 192 32 32
    
        return f