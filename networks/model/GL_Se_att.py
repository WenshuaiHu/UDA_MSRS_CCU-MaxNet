# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:34:25 2024

@author: user
"""

import torch
from torch import nn, einsum
import torch.nn.functional as func

from einops import rearrange
from typing import Optional

    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Group_ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Short_Spel_att_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        # self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
        #                           ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = Group_ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x):
        #x = self.cpe[0](x, size)
        x_cur = self.norm1(x)
        x_cur = self.attn(x_cur)
        x_cur = self.drop_path(x_cur) + x
        
        if self.ffn:
            x_cur = x_cur + self.drop_path(self.mlp(self.norm2(x_cur)))
            
        return x_cur
    
class Global_ChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=8, dropout=0., qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        inner_dim = num_heads * dim
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        
        self.proj = nn.Linear(inner_dim, dim)

    def forward(self, x):
        B, N, C = x.shape      
        
        q = self.to_q(x).view(B, self.num_heads, N, C)
        k = self.to_k(x).view(B, self.num_heads, N, C)
        v = self.to_v(x).view(B, self.num_heads, N, C)
        
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, self.num_heads*C)
        x = self.proj(x)
        return x    
    
    
class Long_Spel_att_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        # self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
        #                           ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = Global_ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        
    def forward(self, x):
        #x = self.cpe[0](x, size)
        x_cur = self.norm1(x)
        x_cur = self.attn(x_cur)
        x_cur = self.drop_path(x_cur) + x
        if self.ffn:
            x_cur = x_cur + self.drop_path(self.mlp(self.norm2(x_cur)))
        
        return x_cur
    
    
class Spel_att_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        # self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
        #                           ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.long_att = Long_Spel_att_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                     ffn=ffn, cpe_act=cpe_act)
        
        self.short_att = Short_Spel_att_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                     ffn=ffn, cpe_act=cpe_act)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = ffn
        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
        self.proj = nn.Linear(dim*2, dim)
        
        
    def forward(self, x):
        
        x1 = self.short_att(x)
        x2 = self.long_att(x) 
        
        x = self.proj(torch.cat((x1, x2), dim=2))+x
        #x = x2 + x1
        #x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
        
        
        
        
        
        
        