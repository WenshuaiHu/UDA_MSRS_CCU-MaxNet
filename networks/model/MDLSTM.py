import torch
from torch import nn, einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import einops
from einops import rearrange
from ..model.patch_merge_expand import PatchMerging, PatchExpanding
#from ..model.GL_Se_att import Short_Spel_att_Block, Long_Spel_att_Block, Spel_att_Block
from ..model.CausalConv import CausalConv1d

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

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout: float):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2 * dim, dim)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, hx=None):
        #H, W = self.input_resolution
        #print(222222, x.shape)
        B, H, W, C = x.shape
        #assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = x.view(B, H * W, C)
        x = shortcut.view(B, H * W, C) + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.view(B, H , W, C)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.ConvT = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                        stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, H, W, C = x.shape
        #assert L == H * W, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.ConvT(x)

        return x
        
class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class SwinTransformerBlocks(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super(SwinTransformerBlocks, self).__init__()
        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=(input_resolution, input_resolution), 
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, xt, hx):

        outputs = []

        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)

            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)

                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        return outputs[-1]


class MDLSTMCell(nn.Module):
                                 
    def __init__(self, dim, input_size, patch_size, memory_multimodal, num_heads, window_size, depth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias=True, drop=0., attn_drop=0., dropout = 0.1, 
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(MDLSTMCell, self).__init__()

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
        self.Mtox =  nn.Linear(memory_multimodal, input_size**2)
        
        self.to_outM = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.memory_attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=num_heads, batch_first=True)
        self.conv1d = CausalConv1d(
                dim=dim,
                kernel_size=4,
                bias=True,
        )
        self.conv_act_fn = nn.SiLU()
    def forward(self, x0, y0, M0):
        
        # Attention Long Memory for multimodal data
        M_ = M0
        x_ = x0
        y_ = y0
        
        
        M_cur = self.conv1d(M0)
        M_cur = self.conv_act_fn(M_cur)

        M_cur = self.to_q(M_cur)# 4 12 64
        x_cur = self.to_k(x0)      # 4 1024 64 
        y_cur = self.to_v(y0)

        M_cur = rearrange(M_cur, 'b n (h d) -> b h n d', h = self.num_heads)
        x_cur = rearrange(x_cur, 'b n (h d) -> b h n d', h = self.num_heads)
        y_cur = rearrange(y_cur, 'b n (h d) -> b h n d', h = self.num_heads)
        
        dots = einsum('b h i d, b h j d -> b h i j', M_cur, x_cur) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        f_t = einsum('b h i j, b h j d -> b h i d', attn, y_cur)### 4 2 12 64   同M_cur
        f_t = rearrange(f_t, 'b h n d -> b n (h d)')
        f_t = self.to_outM(f_t)
        
        
        # Select multimodal information
        F_t = torch.sigmoid(f_t)
        ceil = torch.tanh(f_t)
        # Update Long Memory
        M_next = F_t * (ceil + M_)### 4 64 64   同M_cur Batchsize memory dim
        
        
        T = torch.transpose(self.Mtox(torch.transpose(M_next, 1, 2)), 1,2)
        
        ## Modality update
        x = torch.sigmoid(x_) * torch.tanh(T)### 4 1024 64
        y = torch.sigmoid(y_) * torch.tanh(T)
        
        # M_next = rearrange(M_next, 'b h n d -> b n (h d)')
        # M_next = self.to_outM(M_next)
        # print(222222222222222222222, M_next.shape)

        #x = rearrange(x, 'b h n d -> b n (h d)')
        #x = self.to_outx(x)
        
        #y = rearrange(y, 'b h n d -> b n (h d)')
        #y = self.to_outy(y)

        return x, y, M_next


class MDconvert_down(nn.Module):
    r""" MDconvert

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, index, img_size, patch_size, embed_dim, depths, memory_multimodal, num_heads, window_size,
                 patch_merging = True, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias = True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):

        super(MDconvert_down, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim * 2 ** index
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        
        #self.short_spe_att = Spel_att_Block(self.embed_dim, num_heads=2)
        
        #patches_resolution = self.patch_embed.patches_resolution

        # self.PatchInflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = MDLSTMCell(dim=self.embed_dim,
                                 input_size=img_size,
                                 patch_size = patch_size,
                                 memory_multimodal = memory_multimodal,
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 proj_bias=proj_bias, 
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=drop_path_rate,
                                 norm_layer=norm_layer)
            self.layers.append(layer)
            
        self.Swin = SwinTransformerBlocks(dim=self.embed_dim, input_resolution=patch_size, depth=depths[0],
                                          num_heads=num_heads[0], window_size=window_size, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=drop_path_rate, norm_layer=norm_layer)
                                          
        self.memory_dropout = nn.Dropout(drop_rate)
        self.memory_norm = nn.LayerNorm(self.embed_dim)
        #self.memory_ffn = FFN(d_model=embed_dim, d_ffn=embed_dim*2, dropout=drop_rate)
        
        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
            self.to_outM = nn.Sequential(
                nn.Linear(embed_dim * 2 ** index, self.embed_dim*2),
                #nn.Dropout(dropout)
            )
        else:
            self.downsample = nn.Identity()
            self.to_outM = nn.Identity()

    def forward(self, x, y, M):
        
        #print(111111111111111111111, x.shape)
        #x,_ = self.short_spe_att(x, [int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)])
        #print(222222222222222222222, x.shape)
        
        for index, layer in enumerate(self.layers):
            x, y, M = layer(x, y, M)
        B, L, C = x.shape

        x = rearrange(x, 'B (H W) C -> B H W C', H=int(x.shape[1] ** 0.5))
        y = rearrange(y, 'B (H W) C -> B H W C', H=int(y.shape[1] ** 0.5))
        
        ## Multimodality fusion
        xy = self.Swin(x, y)
        #xy = self.memory_norm(xy)
        #xy = self.memory_ffn(xy)
        
        xy = self.downsample(xy)###
        #xy = rearrange(xy, 'B H W C -> B (H W) C')
        
        x = self.downsample(x)###
        x = rearrange(x, 'B H W C -> B (H W) C')
        
        y = self.downsample(y)###
        y = rearrange(y, 'B H W C -> B (H W) C')
        
        M = self.to_outM(M)
        
        return xy, x, y, M

class MDLSTM_down(nn.Module):
    r""" multimodal Vision LSTM

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
    """

    def __init__(self, index, img_size, patch_size, embed_dim, memory_multimodal, depths, 
                 num_heads, window_size, drop_rate, attn_drop_rate, drop_path_rate, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias = True, 
                 patch_merging = True, norm_layer=nn.LayerNorm):
        super(MDLSTM_down, self).__init__()

        self.ST = MDconvert_down(index = index, img_size = img_size, patch_size=patch_size,
                            embed_dim=embed_dim, memory_multimodal = memory_multimodal, 
                            depths=depths, num_heads=num_heads,  window_size=window_size, 
                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                            drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, qk_scale=qk_scale, proj_bias = proj_bias, 
                            patch_merging = patch_merging, norm_layer = norm_layer)

    def forward(self, inputx, inputy, memmory):

        outputxy, inputx, inputy, memmory = self.ST(inputx, inputy, memmory)

        return outputxy, inputx, inputy, memmory




class MDconvert_up(nn.Module):
    r""" MDconvert

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, index, img_size, patch_size, embed_dim, memory_multimodal, depths, num_heads, window_size,
                 patch_expanding = True, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias = True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):

        super(MDconvert_up, self).__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim * 2 ** index
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        #patches_resolution = self.patch_embed.patches_resolution

        # self.PatchInflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = MDLSTMCell(dim=self.embed_dim,
                                 input_size=img_size,
                                 patch_size = patch_size,
                                 memory_multimodal = memory_multimodal,
                                 depth=depths[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 proj_bias=proj_bias, 
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=drop_path_rate,
                                 norm_layer=norm_layer)
            self.layers.append(layer)
            
        # self.Swin = SwinTransformerBlocks(dim=self.embed_dim, input_resolution=patch_size, depth=depths[0],
        #                                   num_heads=num_heads[0], window_size=window_size, mlp_ratio=mlp_ratio,
        #                                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
        #                                   drop_path=drop_path_rate, norm_layer=norm_layer)
                                          
        #self.memory_dropout = nn.Dropout(drop_rate)
        #self.memory_norm = nn.LayerNorm(self.embed_dim)
        #self.memory_ffn = FFN(d_model=embed_dim, d_ffn=embed_dim*2, dropout=drop_rate)
        
        if patch_expanding:
            self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
            self.to_outM = nn.Sequential(
                nn.Linear(embed_dim * 2 ** index, self.embed_dim//2),
                #nn.Dropout(dropout)
            )
        else:
            self.upsample = nn.Identity()
            self.to_outM = nn.Identity()

    def forward(self, x, y, M):

        for index, layer in enumerate(self.layers):
            x, y, M = layer(x, y, M)
        B, L, C = x.shape

        x = rearrange(x, 'B (H W) C -> B H W C', H=int(x.shape[1] ** 0.5))
        y = rearrange(y, 'B (H W) C -> B H W C', H=int(y.shape[1] ** 0.5))
        
        x = self.upsample(x)###
        x = rearrange(x, 'B H W C -> B (H W) C')
        
        y = self.upsample(y)###
        y = rearrange(y, 'B H W C -> B (H W) C')
        
        M = self.to_outM(M)
        
        return x, y, M

class MDLSTM_up(nn.Module):
    r""" multimodal Vision LSTM

    Args:
        img_size (int | tuple(int)): Input image size.
        patch_size (int | tuple(int)): Patch size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
    """

    def __init__(self, index, img_size, patch_size, embed_dim, memory_multimodal, depths, 
                 num_heads, window_size, drop_rate, attn_drop_rate, drop_path_rate, 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_bias = True, 
                 patch_expanding = True, norm_layer=nn.LayerNorm):
        super(MDLSTM_up, self).__init__()

        self.ST = MDconvert_up(index = index, img_size = img_size, patch_size=patch_size, 
                            embed_dim=embed_dim, memory_multimodal=memory_multimodal,
                            depths=depths, num_heads=num_heads,  window_size=window_size, 
                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                            drop_path_rate=drop_path_rate, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, qk_scale=qk_scale, proj_bias = proj_bias, 
                            patch_expanding = patch_expanding, norm_layer = norm_layer)

    def forward(self, inputx, inputy, memmory):

        inputx, inputy, memmory = self.ST(inputx, inputy, memmory)

        return inputx, inputy, memmory