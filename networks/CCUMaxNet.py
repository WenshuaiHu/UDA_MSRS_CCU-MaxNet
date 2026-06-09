import os
import logging

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
#from utils.highDHA_utils import initialize_weights
from .model.models_MAE_xLSTM import MaskedAutoencoderViT, Swin_MAE_Segmenter
from einops import rearrange

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
        
class CCUMaxNet(nn.Module):

    def __init__(self, band, patchsize, num_classes, num_channels = 128):
        super(CCUMaxNet, self).__init__()
        
        self.patchsize = patchsize
        # stem net for hsi
        self.conv1 = nn.Conv2d(band, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        #self.relu = nn.ReLU(inplace=False)
        self.relu = hswish()
        # stem net for msi
        self.conv_msi = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_msi = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        # stem net for sar
        #self.conv_sar = nn.Conv2d(2, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn_sar = nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM)
        
        self.MAE_dim = 64
        self.patch_size = [2, 8]
        ## MIM的输入均是降维64x64x128   32 32 64   16 16 128   8 8 256
        self.Swin_MAE = MaskedAutoencoderViT(img_size=(patchsize, patchsize), 
                        in_chans=num_channels, patch_size=self.patch_size, 
                        embed_dim=self.MAE_dim, decoder_embed_dim=self.MAE_dim, 
                        mlp_ratio=4., window_size = 4) ## SwinT，用于输出融合
                        
        self.Swin_MAE_Segmenter = Swin_MAE_Segmenter(self.Swin_MAE.Encoder, 
                        dim = self.MAE_dim)
        
        out_channel = 64 #self.MAE_dim
        self.convx1 = nn.Conv2d(self.MAE_dim, out_channel, 1, 1)              
        self.convx2 = nn.Conv2d(self.MAE_dim*2, out_channel, 1, 1)
        self.convx3 = nn.Conv2d(self.MAE_dim*4, out_channel, 1, 1)
   
        self.fuse_out = (out_channel)*3
        self.relu_fuse = hswish()
        
        self.transconv = nn.Sequential(
            nn.Conv2d(self.fuse_out, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(self.fuse_out, 512, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            #nn.ConvTranspose2d(self.fuse_out, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            #nn.ReLU(inplace=False),
            hswish()
            # nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Conv2d(128, num_classes, 1, 1)
        
        self.tanh = nn.Tanh()
    
    def forward(self, x, y, domain):
        _, _, height, width = x.shape #? 10 128 128

        ## 维度对齐
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn_msi(self.conv_msi(y)))
        
        ## 重构与特征提取
        
        loss, pred_x, pred_y = self.Swin_MAE(x, y, mask_ratio=0.75)
        feature_fuse_list, latentx, latenty, M_curr_list, xytemp1 = self.Swin_MAE_Segmenter(x, y)# ? 8 8 384*3
        
        x0_h = self.patchsize//self.patch_size[0]
        x0_w = self.patchsize//self.patch_size[0]
        
        xy1 = feature_fuse_list[0]
        xy1 = self.convx1(xy1)
        
        xy2 = F.interpolate(feature_fuse_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        xy2 = self.convx2(xy2)
        
        xy3 = F.interpolate(feature_fuse_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        xy3 = self.convx3(xy3)
        
        outputs = [] 
        outputs.append(xy1)
        outputs.append(xy2)
        outputs.append(xy3)
        
        xt = self.relu_fuse(torch.cat(outputs, 1))## 4 2016 32 32
        
            
        ### head
        out = F.interpolate(xt, size=(height, width), mode='bilinear', align_corners=True)
        out = self.transconv(out)
        out = self.final_conv(out)
        
        #return x, refine_x, refine_y, refine_z, out, loss
        return xt, latentx, latenty, out, loss, M_curr_list, xytemp1