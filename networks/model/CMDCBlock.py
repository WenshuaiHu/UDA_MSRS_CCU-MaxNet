# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:05:36 2024

@author: user
"""

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
from ..model.GL_Se_att import Spel_att_Block
from ..Fractional_Gabor_2D_Conv_layer.F_gabor_2Dconv_layer import F_gabor_conv
#from ..Fractional_Gabor_2D_Conv_layer.opsfrac import gen_gf_list

class CMDCBlock(nn.Module):

    def __init__(self, dim=32, num_heads=2):
        super().__init__()
        
        self.avg1 = nn.AvgPool2d((1, 1))
        self.avg2 = nn.AvgPool2d((1, 1))
        self.short_spe_att_spec = Spel_att_Block(dim, num_heads=num_heads)
        
        self.f = [1., 1./2, 1./3, 1./4]
        self.dim = dim
        
        self.conv_size = 3
        fractional_order = 25
        # filter_size_1 = [conv_size, conv_size, 64, 16]
        self.order = fractional_order*0.01  ## fractional order   
        
        self.conv1 = nn.Conv2d(dim*4, dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        
    def forward(self, x, y):
        
        x_ = self.short_spe_att_spec(x)
        
        H = W = int(x.shape[1] ** 0.5) 
        x0 = rearrange(x, 'B (H W) C  -> B C H W', H =H, W=W)
        y0 = rearrange(y, 'B (H W) C  -> B C H W', H =H, W=W)
        
        y_ = F_gabor_conv(y0, self.f, self.dim, kernel_size=self.conv_size, order = self.order, bias=None, stride=1, padding=1, dilation=1, groups=1)
        y_ = self.conv1(y_)   
        y_ = rearrange(y_, 'B C H W  -> B (H W) C')
        
        # H = W = int(x.shape[1] ** 0.5) 
        d_xy = x0 - y0
        f_gap_xy = torch.sigmoid(self.avg1(d_xy))
        
        x_temp2_y = f_gap_xy * x0
        y1 = x_temp2_y + y0
        
        d_yx = y0 - x0
        f_gap_yx = torch.sigmoid(self.avg2(d_yx))
        y_temp2_x = f_gap_yx * y0
        x1 = y_temp2_x + x0
        
        x1 = rearrange(x1, 'B C H W  -> B (H W) C')
        y1 = rearrange(y1, 'B C H W  -> B (H W) C')

        x = x + x1 + x_
        y = y + y1 + y_
        
        return x, y, [x_temp2_y, y_temp2_x]
    
    
    
    
    
    
    
    
    
    
    
    
    