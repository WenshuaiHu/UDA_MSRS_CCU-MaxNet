# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import logging
import pdb
import numpy as np
from einops import rearrange
from enum import Enum
import einops

#from timm.models.vision_transformer import Block
from ..model.utils.pos_embed import get_2d_sincos_pos_embed

#from ..model.xLSTM.seglstm import ViLBlock
from ..model.patch_merge_expand import Cross_scale_PatchEmbed
from ..model.MDLSTM import MDLSTM_down, MDLSTM_up, SwinTransformerBlocks
from ..model.GL_Se_att import Spel_att_Block
# copied from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py
from ..model.CMDCBlock import CMDCBlock
       
class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"
    
class Swin_MAE_Encoder(nn.Module):
    """ 
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3,
                 memory_multimodal = 12,
                 embed_dim=1024, depth=(2, 2, 6, 2), num_heads=(2, 6, 12, 24), 
                 decoder_depth=(2, 2), decoder_num_heads=(6, 12), 
                 window_size = 8, norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
       ## modality HSI
        #self.patch_embed_x = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_x = Cross_scale_PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        num_patches = self.patch_embed_x.num_patches

        #self.cls_token_x = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        torch.nn.init.normal_(self.pos_embed_x, std=.02)


        ## modality MSI
        #self.patch_embed_y = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_y = Cross_scale_PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches
        #self.cls_token_y = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        torch.nn.init.normal_(self.pos_embed_y, std=.02)



        ## modality SAR
        #self.patch_embed_z = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.cls_token_z = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed_z = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.mask_token_x = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_y = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.mask_token_z = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        '''
        Encoder Part  patch降维，通道数升维
        '''
        encoder_block = 2
        #self.norm = norm_layer(embed_dim*encoder_block)
        # 20240623 xLSTM
        
        # self.depth = 6
        # self.depth1 = [2, 2, 2]
        # self.alternation = "bidirectional"
        # self.drop_path_rate = 0.0
        # self.drop_path_decay = False
        self.patch_merging = True
        
        # # calculate stochastic depth per block
        # if self.drop_path_decay and self.drop_path_rate > 0.:
        #     dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # else:
        #     dpr = [self.drop_path_rate] * self.depth

        # # directions
        # directions = []
        # if self.alternation == "bidirectional":
        #     for i in range(self.depth):
        #         if i % 2 == 0:
        #             #directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        #             directions.append("rowwise_from_top_left")
        #         else:
        #             #directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        #             directions.append("rowwise_from_bot_right")
        # else:
        #     raise NotImplementedError(f"invalid alternation '{alternation}'")

        # blocks specific
        # self.blocks_x = nn.ModuleList(
        #     [
        #         ViLBlock(
        #             dim=embed_dim,
        #             drop_path=dpr[i],
        #             direction=directions[i],
        #         )
        #         for i in range(self.depth1[0])
        #     ]
        # )
        # self.blocks_y = nn.ModuleList(
        #     [
        #         ViLBlock(
        #             dim=embed_dim,
        #             drop_path=dpr[i + (self.depth//len(self.depth1))],
        #             direction=directions[i + (self.depth//len(self.depth1))],
        #         )
        #         for i in range(self.depth1[1])
        #     ]
        # )
        Encoder_spec_layer = 2
        self.blocks_spec = self.build_layers_speci(depths = depth, embed_dim=embed_dim, img_size = img_size, patch_size = patch_size[0], window_size = window_size, 
                                                num_heads = num_heads, memory_multimodal = memory_multimodal, num_layers = Encoder_spec_layer)
        
        self.short_spe_att_spec = Spel_att_Block(embed_dim, num_heads=8)
        self.norm_x = norm_layer(embed_dim*Encoder_spec_layer)
        self.norm_y = norm_layer(embed_dim*Encoder_spec_layer)
        
        
        
        self.Swin = SwinTransformerBlocks(dim=embed_dim, input_resolution=patch_size[0], depth=depth[0],
                                          num_heads=num_heads[0], window_size=window_size)
        #self.memory_norm =  nn.LayerNorm(embed_dim)
        
        self.M_curr = nn.Parameter(torch.empty(1, memory_multimodal, embed_dim), requires_grad=True)  # Multimodal memory token 
        torch.nn.init.xavier_normal_(self.M_curr)## 正态分布
        self.feature_fuse_list = []
        
        
        Encoder_share_layer = 2   ## 因为self.blocks中降维一次，因此，img_size 变为 img_size[0]//patch_size//2
        self.blocks_share = self.build_layers_share(embed_dim=embed_dim*2, img_size = img_size[0]//patch_size[0]//2, patch_size = patch_size[0], 
                                              window_size = window_size, depths = decoder_depth, num_heads = decoder_num_heads, 
                                              memory_multimodal = memory_multimodal, 
                                              num_layers = Encoder_share_layer)
                                              # 不论几层，只在最后一层通道数增加一倍  
        self.norm_xs = norm_layer(embed_dim*Encoder_spec_layer*Encoder_share_layer)
        self.norm_ys = norm_layer(embed_dim*Encoder_spec_layer*Encoder_share_layer)
        
        #self.short_spe_att_share = Spel_att_Block(embed_dim*2, num_heads=8)
        #self.norm = norm_layer(embed_dim*Encoder_spec_layer*Encoder_share_layer)
        
        # if self.patch_merging:
            # self.downsample_x = PatchMerging(dim=embed_dim, norm_layer=nn.LayerNorm)
            # self.downsample_y = PatchMerging(dim=embed_dim, norm_layer=nn.LayerNorm)
        # else:
            # self.downsample_x = nn.Identity()
            # self.downsample_y = nn.Identity()
            
        # blocks share
        # self.blocks_s = nn.ModuleList(
        #     [
        #         ViLBlock(
        #             dim=embed_dim*encoder_block,
        #             drop_path=dpr[i+(self.depth//len(self.depth1))*2],
        #             direction=directions[i+(self.depth//len(self.depth1))*2],
        #         )
        #         for i in range(self.depth1[2])
        #     ]
        # )
        # if self.patch_merging:
            # self.downsample_s = PatchMerging(dim=embed_dim*2, norm_layer=nn.LayerNorm)
            # self.downsample_s = PatchMerging(dim=embed_dim*2, norm_layer=nn.LayerNorm)
        # else:
            # self.downsample_s = None
            # self.downsample_s = None
            
            
            
            
        self.dynamic_cross_modal_enhance1 = CMDCBlock(dim = embed_dim, num_heads = 8)
        # self.dynamic_cross_modal_enhance2 = CMDCBlock()
        self.dynamic_cross_modal_enhance3 = CMDCBlock(dim = embed_dim*2, num_heads = 8)
        
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed_x = get_2d_sincos_pos_embed(self.pos_embed_x.shape[-1], self.patch_embed_x.grid_size, cls_token=False)  # modified
        self.pos_embed_x.data.copy_(torch.from_numpy(pos_embed_x).float().unsqueeze(0))
        ##
        pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], self.patch_embed_y.grid_size, cls_token=False)  # modified
        self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))
        ##        
        #pos_embed_z = get_2d_sincos_pos_embed(self.pos_embed_z.shape[-1], self.patch_embed_z.grid_size, cls_token=True)  # modified
        #self.pos_embed_z.data.copy_(torch.from_numpy(pos_embed_z).float().unsqueeze(0))
        
        #w = self.patch_embed_x.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        #w = self.patch_embed_y.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        #w = self.patch_embed_z.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        
        #torch.nn.init.normal_(self.cls_token_x, std=.02)
        torch.nn.init.normal_(self.mask_token_x, std=.02)
        
        #torch.nn.init.normal_(self.cls_token_y, std=.02)
        torch.nn.init.normal_(self.mask_token_y, std=.02)
        
        #torch.nn.init.normal_(self.cls_token_z, std=.02)
        #torch.nn.init.normal_(self.mask_token_z, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def window_masking(self, x: torch.Tensor, y: torch.Tensor, r: int = 4, flag: int = 0, mask_ratio: float = 0.75, 
                       remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together

        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x

        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image

        x: [N, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        #x = rearrange(x, 'B H W C -> B (H W) C')
        B, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)

        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - mask_ratio))]

        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)

        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0) 
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=np.int_) 
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)

        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        if flag ==0:
            mask_token = self.mask_token_x
        else:# flag ==1:
            mask_token = self.mask_token_y
        #else:
            #mask_token = self.mask_token_z
            
        if remove:
            x_masked = torch.gather(x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            return x_masked, mask, sparse_restore
        else:
            x_masked = torch.clone(x)
            for i in range(B):   
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
            x_masked = rearrange(x_masked, 'B (H W) C -> B H W C', H=int(x_masked.shape[1] ** 0.5))
            
            y_masked = torch.clone(y)
            for i in range(B):   
                y_masked[i, index_mask.cpu().numpy()[i, :], :] = mask_token
            y_masked = rearrange(y_masked, 'B (H W) C -> B H W C', H=int(y_masked.shape[1] ** 0.5))
            
            return x_masked, y_masked, mask
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_tokenizer(self, x, y, mask_ratio):
        # embed patches
        x = self.patch_embed_x(x)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        x = x + self.pos_embed_x###维度  4 256 720
        # masking: length -> length * mask_ratio
        #x, mask_x, ids_restore_x = self.random_masking(x, mask_ratio)
        # x, mask_x = self.window_masking(x, r = 4, flag = 0, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        # append cls token
        # cls_token_x = self.cls_token_x + self.pos_embed_x[:, :1, :]
        # cls_tokens_x = cls_token_x.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens_x, x), dim=1)

        y = self.patch_embed_y(y)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        y = y + self.pos_embed_y###维度  4 256 720
        # print(1133, y.shape)  
        # masking: length -> length * mask_ratio
        #y, mask_y, ids_restore_y = self.random_masking(y, mask_ratio)
        # y, mask_y = self.window_masking(y, r = 4, flag = 1, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        
        x, y, mask_x = self.window_masking(x, y, r = 4, flag = 0, mask_ratio = mask_ratio, remove=False, mask_len_sparse=False)# B H W C
        
        # append cls token
        # cls_token_y = self.cls_token_y + self.pos_embed_y[:, :1, :]
        # cls_tokens_y = cls_token_y.expand(y.shape[0], -1, -1)
        # y = torch.cat((cls_tokens_y, y), dim=1)###维度 4 65 720
        # print(1122, y.shape)  
        
        '''
        z = self.patch_embed_z(z)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        z = z + self.pos_embed_z[:, 1:, :]
        # masking: length -> length * mask_ratio
        #z, mask_z, ids_restore_z = self.random_masking(z, mask_ratio)
        z, mask_z = self.window_masking(z, r = 4, flag = 2, remove=False, mask_len_sparse=False)# B H W C'''
        # append cls token
        # cls_token_z = self.cls_token_z + self.pos_embed_z[:, :1, :]
        # cls_tokens_z = cls_token_z.expand(z.shape[0], -1, -1)
        # z = torch.cat((cls_tokens_z, z), dim=1)
        #print(0, x.shape, y.shape, z.shape)
        ## 16 16 64*4*4
        return x, y, mask_x, mask_x#, ids_restore_x, ids_restore_y, ids_restore_z
        
    def build_layers_speci(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, img_size = 32, patch_size = 4, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, qkv_bias: bool = True, proj_bias: bool = True,  mlp_ratio: float = 4., drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0., 
                            num_layers: int = 1, memory_multimodal: int = 12, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        ### 当 patch_merging是True即i=0时，输入空间大小img_size不变，但此时输出大小已减半。所以，当False即i=1时，img_size需要减半
        for i in range(num_layers):
            layer = MDLSTM_down(
                index=i,
                img_size = img_size[0]//patch_size//2 if i == num_layers - 1 else img_size[0]//patch_size, 
                patch_size = patch_size, 
                embed_dim = embed_dim, 
                memory_multimodal = memory_multimodal,
                depths = depths, 
                num_heads = num_heads, 
                window_size = window_size, 
                drop_rate = drop_rate, 
                attn_drop_rate = attn_drop_rate, 
                drop_path_rate = drop_path_rate, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                proj_bias = proj_bias, 
                patch_merging = False if i == num_layers - 1 else True, 
                norm_layer=norm_layer)
            self.img_size_1 = img_size[0]//patch_size//2 if i == num_layers - 1 else img_size[0]//patch_size
            layers.append(layer)
        
        return layers  ### 维度对齐
    def build_layers_share(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, img_size = 32, patch_size = 4, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, qkv_bias: bool = True, proj_bias: bool = True,  mlp_ratio: float = 4., drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0., 
                            num_layers: int = 1, memory_multimodal: int = 12, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        ### 当 patch_merging是True即i=0时，输入空间大小img_size不变，但此时输出大小已减半。所以，当False即i=1时，img_size需要减半
        for i in range(num_layers):
            layer = MDLSTM_down(
                index=i,
                img_size = img_size//2 if i == num_layers - 1 else img_size, 
                patch_size = patch_size, 
                embed_dim = embed_dim, 
                memory_multimodal = memory_multimodal,
                depths = depths, 
                num_heads = num_heads, 
                window_size = window_size, 
                drop_rate = drop_rate, 
                attn_drop_rate = attn_drop_rate, 
                drop_path_rate = drop_path_rate, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                proj_bias = proj_bias, 
                patch_merging = False if i == num_layers - 1 else True, 
                norm_layer=norm_layer)
            self.img_size_2 = img_size//2 if i == num_layers - 1 else img_size
            layers.append(layer)
        
        return layers  ### 维度对齐     
    
    def forward_encoder_specific(self, x, y):## 维度对齐  输入：B H W C
        

        #xy0 = self.memory_norm(xy0)
        # self.feature_fuse_list.append(xy0)
        
        B, _, _, _ = x.shape
        M_curr = self.M_curr
        M_curr = M_curr.repeat(B, 1, 1)### 通道维度 重复
        
        x = rearrange(x, 'B H W C -> B (H W) C')
        y = rearrange(y, 'B H W C -> B (H W) C')
        
        x0 = self.short_spe_att_spec(x)
        
        H = W = int(x.shape[1] ** 0.5) 
        x0 = rearrange(x0, 'B (H W) C  -> B H W C', H =H, W=W)
        y0 = rearrange(y, 'B (H W) C  -> B H W C', H =H, W=W)
        xy0 = self.Swin(x0, y0)  ### 4 32 32 64
        
        x, y, [x_temp2_y1, y_temp2_x1] = self.dynamic_cross_modal_enhance1(x, y)

        # apply blocks1
        for block in self.blocks_spec:
            xy1, x, y, M_curr = block(x, y, M_curr)
        x = self.norm_x(x)
        y = self.norm_y(y)

        #x, y = self.dynamic_cross_modal_enhance2(x, y)
        
        # x = rearrange(x, 'B (H W) C -> B H W C', H=self.img_size_1)
        # y = rearrange(y, 'B (H W) C -> B H W C', H=self.img_size_1)
        # self.feature_fuse_list.append(xy)

        return xy0, xy1, x, y, M_curr, [x_temp2_y1, y_temp2_x1]## 结构顺序：SwinT path merge SwinT   降维一次   4 16 16 128
        
    def forward_encoder_share(self, x, y, M_curr):
        
        # x = self.short_spe_att_share(x)
        x, y, [x_temp2_y2, y_temp2_x2] = self.dynamic_cross_modal_enhance3(x, y)
        
        for block in self.blocks_share:
            xy2, x, y, M_curr= block(x, y, M_curr)
        x = self.norm_xs(x)
        y = self.norm_ys(y)
            

            
        # x = rearrange(x, 'B (H W) C -> B H W C', H=self.img_size_2)
        # y = rearrange(y, 'B (H W) C -> B H W C', H=self.img_size_2)
        # self.feature_fuse_list.append(xy)
        
        return xy2, x, y, M_curr, [x_temp2_y2, y_temp2_x2]## 结构顺序：SwinT path merge SwinT  降维一次


    def forward(self, imgsx, imgsy, mask_ratio):

        latentx, latenty, mask_x, mask_y = self.forward_tokenizer(imgsx, imgsy, mask_ratio)## 4 32 32 64
        
        M_curr_list = []
        xy0, xy1, x, y, M_curr, xytemp1 = self.forward_encoder_specific(latentx, latenty)
        M_curr_list.append(M_curr)
        xy2, x, y, M_curr, xytemp2 = self.forward_encoder_share(x, y, M_curr)
        M_curr_list.append(M_curr)
                
        return x, y, M_curr_list, xytemp1, xytemp2, mask_x, mask_y## 降维2次   4 8 8 256



class Swin_MAE_Dncoder(nn.Module):
    """ 
    """
    def __init__(self, imagesize, patch_size = 16, in_chans = 3, decoder_embed_dim=512, memory_multimodal = 12, encoder_block = 2,
                 decoder_depth=(2, 2), decoder_num_heads=(6, 12),
                 mlp_ratio=4., window_size = 8, norm_layer=nn.LayerNorm):
        super().__init__()

        Encoder_number = 2 ## build_layers_speci + build_layers_share  增加一个Encoder，通道数增加一倍

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        '''
        Decoder Part  patch升维，通道数降维
        '''
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # self.depth = 2
        # #self.depth1 = [2,2]
        # self.alternation = "bidirectional"
        # self.drop_path_rate = 0.0
        # self.drop_path_decay = False
        # self.patch_expanding = True
        
        # calculate stochastic depth per block
        # if self.drop_path_decay and self.drop_path_rate > 0.:
        #     dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # else:
        #     dpr = [self.drop_path_rate] * self.depth

        # # directions
        # directions = []
        # if self.alternation == "bidirectional":
        #     for i in range(self.depth):
        #         if i % 2 == 0:
        #             #directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        #             directions.append("rowwise_from_top_left")
        #         else:
        #             #directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        #             directions.append("rowwise_from_bot_right")
        # else:
        #     raise NotImplementedError(f"invalid alternation '{alternation}'")
        
        # self.blocks_up = nn.ModuleList(
        #     [
        #         ViLBlock(
        #             dim=decoder_embed_dim*encoder_block,
        #             drop_path=dpr[i],
        #             direction=directions[i],
        #         )
        #         for i in range(self.depth)
        #     ]
        # )
        
        decoder_share= 2
        ## 由于编码器部分降维了2次，所以输入通道数应该是64的4倍输入分辨率同编码器，8
        self.blocks_up_share = self.build_decoder_share(depths = decoder_depth, embed_dim=decoder_embed_dim*2**2, memory_multimodal = memory_multimodal,
                                                  img_size = imagesize, patch_size = patch_size, window_size = window_size, 
                                                  num_heads = decoder_num_heads, num_layers = decoder_share)  
        self.decoder_norm_x = norm_layer(decoder_embed_dim*2)
        self.decoder_norm_y = norm_layer(decoder_embed_dim*2)
        
        
        ## 由于共享解码器已经上采样1次，输入通道数应该是64的2倍，输入分辨率增大一倍，16
        self.blocks_up_spec = self.build_decoder_spec(depths = decoder_depth, embed_dim=decoder_embed_dim*2, memory_multimodal = memory_multimodal,
                                                  img_size = imagesize*2, patch_size = patch_size, window_size = window_size, 
                                                  num_heads = decoder_num_heads, num_layers = decoder_share)  
        
        self.decoder_norm_xs = norm_layer(decoder_embed_dim)
        self.decoder_norm_ys = norm_layer(decoder_embed_dim)
        
        patch_expanding = True
        ## encoder_block 数对应下采样次数，对应编码器部分通道数的倍数
        #self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim * encoder_block**2, norm_layer=norm_layer)
        # if patch_expanding:
            # self.upsample_up = PatchExpanding(dim=decoder_embed_dim*encoder_block, norm_layer=norm_layer)
        # else:
            # self.upsample_up = nn.Identity()
        

        '''
        Decoder Part  跨模态信息交互，patch维度恢复
        '''
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
           
        self.decoder_pred_x = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_pred_y = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        #self.decoder_pred_z = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)  # modified
        # print(pos_embed.shape)
        # print(self.pos_embed.shape)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def build_decoder_share(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, memory_multimodal = 12, img_size = 32, patch_size = 4, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, qkv_bias: bool = True, proj_bias: bool = True,  mlp_ratio: float = 4., 
                            drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0., 
                            num_layers: int = 1, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        ### 当 patch_merging是True即i=0时，输入空间大小img_size不变，但此时输出大小已减半。所以，当False即i=1时，img_size需要减半
        for i in range(num_layers-1):
            layer = MDLSTM_up(
                index=i,
                img_size = img_size if i == num_layers - 2 else img_size*2, 
                patch_size = patch_size, 
                embed_dim = embed_dim, 
                memory_multimodal = memory_multimodal,
                depths = depths, 
                num_heads = num_heads, 
                window_size = window_size, 
                drop_rate = drop_rate, 
                attn_drop_rate = attn_drop_rate, 
                drop_path_rate = drop_path_rate, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                proj_bias = proj_bias, 
                patch_expanding = True if i == num_layers - 2 else False, 
                norm_layer=norm_layer)
            layers.append(layer)
        return layers  ### 维度对齐    
    
    def build_decoder_spec(self, depths: tuple = (2, 2, 6, 2), embed_dim=96, memory_multimodal = 12, img_size = 32, patch_size = 4, 
                            num_heads: tuple = (2, 6, 12, 24), drop_path: float = 0.1, 
                            window_size: int = 8, qkv_bias: bool = True, proj_bias: bool = True,  mlp_ratio: float = 4., 
                            drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0., 
                            num_layers: int = 1, norm_layer=nn.LayerNorm):
        layers = nn.ModuleList()
        ### 当 patch_merging是True即i=0时，输入空间大小img_size不变，但此时输出大小已减半。所以，当False即i=1时，img_size需要减半
        for i in range(num_layers-1):
            layer = MDLSTM_up(
                index=i,
                img_size = img_size if i == num_layers - 2 else img_size*2, 
                patch_size = patch_size, 
                embed_dim = embed_dim, 
                memory_multimodal = memory_multimodal,
                depths = depths, 
                num_heads = num_heads, 
                window_size = window_size, 
                drop_rate = drop_rate, 
                attn_drop_rate = attn_drop_rate, 
                drop_path_rate = drop_path_rate, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                proj_bias = proj_bias, 
                patch_expanding = True if i == num_layers - 2 else False, 
                norm_layer=norm_layer)
            layers.append(layer)
        return layers  ### 维度对齐  
    
            
    def forward_decoder_share(self, x, y, M_curr):
       
        for layer in self.blocks_up_share:
            x, y, M_curr = layer(x, y, M_curr)
        x = self.decoder_norm_x(x)
        y = self.decoder_norm_y(y)
        
        return x, y, M_curr##先升维，后MAE（MAE结构顺序：SwinT+patch expand (再升维)+SwinT） 该部分升维2次  4 256 128 4 16 16 128
        
    def forward_decoder_speci(self, x, y, M_curr, xytemp1):
    
        # predictor projection
        # inp_x = x
        # inp_y = y
        for layer in self.blocks_up_spec:
            x, y, M_curr = layer(x, y, M_curr)    # 4 1024 64  4 32 32 64
        x = self.decoder_norm_xs(x)
        y = self.decoder_norm_ys(y)
            
        x_temp2_y = rearrange(xytemp1[0], 'B C H W  -> B (H W) C')
        y_temp2_x = rearrange(xytemp1[1], 'B C H W  -> B (H W) C')
        
        x = x - y_temp2_x
        y = y - x_temp2_y
            
        
        #x = self.CMA_x(x, inp_x, inp_y)
        x = self.decoder_pred_x(x)               # 4 124 512   4 64 64 128
        
        #y = self.CMA_y(y, inp_x, inp_y)
        y = self.decoder_pred_y(y)              
        
        return x, y## 维度不变
           
    def forward(self, x, y, M_curr_list, temp1):
        # print(2222222222,x.shape, y.shape)
        M_curr_1 = M_curr_list[1]
        x, y, M_up= self.forward_decoder_share(x, y, M_curr_1)
        
        M_curr_0 = M_curr_list[0] + M_up
        x, y= self.forward_decoder_speci(x, y, M_curr_0, temp1)
    
        
        
        return x, y## 4 32 32 96  升维2次，恢复输入分辨率

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=(1, 1), num_heads=(2, 2), 
                 decoder_embed_dim=512, decoder_depth=(1, 1), decoder_num_heads=(2, 2), ####  decoder_num_heads得取值和输入通道数有关，除不尽会报错
                 mlp_ratio=4., window_size = 8, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_size = patch_size[0]
        
        memory_multimodal = embed_dim## 多模态互补信息变量token，维度默认为(dim dim)
        
        
        self.Encoder = Swin_MAE_Encoder(img_size=img_size, patch_size=patch_size, 
                 in_chans=in_chans, embed_dim=embed_dim, memory_multimodal = memory_multimodal, depth=depth, num_heads=num_heads, 
                 decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads, 
                 window_size = window_size, norm_layer=norm_layer)
        
        ## 因为编码器patchsize后，降维2次，输入分辨率降低8倍
        self.Dncoder = Swin_MAE_Dncoder(img_size[0]//(patch_size[0]*2**2), patch_size = patch_size[0], in_chans = in_chans, 
                 decoder_embed_dim=decoder_embed_dim, memory_multimodal = memory_multimodal, decoder_depth=decoder_depth, 
                 encoder_block = 2,
                 decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, 
                 window_size = window_size, norm_layer=norm_layer)
                 
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)  # modified
        # print(pos_embed.shape)
        # print(self.pos_embed.shape)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        #x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        #x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        H = imgs.shape[2]
        W = imgs.shape[3]
        self.patch_info = (H, W, p, h, w)
        
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        #x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        #imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs
     
        
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgsx, imgsy, mask_ratio=0.75):
        
        #print(0, imgsx.shape, imgsy.shape)   ### 4 128 64 64
        latentx, latenty, M_curr_list, xytemp1, xytemp2, mask_x, mask_y = self.Encoder(imgsx, imgsy, mask_ratio)
        pred_x, pred_y  = self.Dncoder(latentx, latenty, M_curr_list, xytemp1)
        
        loss_x = self.forward_loss(imgsx, pred_x, mask_x)
        loss_y = self.forward_loss(imgsy, pred_y, mask_y)
       
        H, W, p, h, w = self.patch_info
        
        loss = (loss_x + loss_y)*0.5
        
        return loss, pred_x, pred_y

class Swin_MAE_Segmenter(torch.nn.Module):
    def __init__(self, encoder : Swin_MAE_Encoder, dim = 96) -> None:
        super().__init__()

        self.forward_encoder_specific = encoder.forward_encoder_specific
        self.forward_encoder_share = encoder.forward_encoder_share
        
        self.patch_embed_x = encoder.patch_embed_x
        self.pos_embed_x   = encoder.pos_embed_x
        self.patch_embed_y = encoder.patch_embed_y
        self.pos_embed_y   = encoder.pos_embed_y
        
        # self.patch_embed_z = encoder.patch_embed_z
        # self.pos_embed_z   = encoder.pos_embed_z
        #self.M_curr1 = nn.Parameter(torch.empty(1, dim, dim*2), requires_grad=True)  # Multimodal memory token 
        #torch.nn.init.xavier_normal_(self.M_curr1)## 正态分布
        #self.M_curr2 = nn.Parameter(torch.empty(1, dim, dim*4), requires_grad=True)  # Multimodal memory token 
        #torch.nn.init.xavier_normal_(self.M_curr2)## 正态分布
        
    def forward(self, imgsx, imgsy):
        
        x = self.patch_embed_x(imgsx)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        x = x + self.pos_embed_x

        y = self.patch_embed_y(imgsy)
        #print("mae的positional是："+str(x.shape))
        # add pos embed w/o cls token
        y = y + self.pos_embed_y
        
        x = rearrange(x, 'B (H W) C -> B H W C', H=int(x.shape[1] ** 0.5))
        y = rearrange(y, 'B (H W) C -> B H W C', H=int(y.shape[1] ** 0.5))
        feature_fuse_list = []
        M_curr_list = []
        
        B, _, _, _ = imgsx.shape
        # M_curr1 = self.M_curr1
        # M_curr1 = M_curr1.repeat(B, 1, 1)
        # M_curr2 = self.M_curr2
        # M_curr2 = M_curr2.repeat(B, 1, 1)
        
        
        xy0, xy1, x, y, M_curr, xytemp1= self.forward_encoder_specific(x, y)# 4 16 16 192
        M_curr_list.append(M_curr)
        # M_curr_list.append(M_curr1)
        xy2, latentx, latenty, M_curr, xytemp2 = self.forward_encoder_share(x, y, M_curr) 
        M_curr_list.append(M_curr)
        # M_curr_list.append(M_curr2)
        
        xy0 = rearrange(xy0, 'B H W C -> B C H W')
        xy1 = rearrange(xy1, 'B H W C -> B C H W')
        xy2 = rearrange(xy2, 'B H W C -> B C H W')
        
        feature_fuse_list.append(xy0)
        feature_fuse_list.append(xy1)
        feature_fuse_list.append(xy2)

        return feature_fuse_list, latentx, latenty, M_curr_list, xytemp1 