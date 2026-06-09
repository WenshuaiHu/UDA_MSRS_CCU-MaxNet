import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init
#from torchsummary import summary
# from models.quaternion_layers import QuaternionLinear
# from models.utils_3D.former import FeedForward
import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class CMAttention(nn.Module):
    def __init__(self, dim, patchsize, heads = 8, dim_head = 64, dropout = 0.1, memory_num = 3):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = True)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = True)

        if patchsize == 128:
            self.fuse_att = nn.Linear(2048+memory_num, 1024+memory_num, bias = True)### 128*128
        elif patchsize == 64: 
            self.fuse_att = nn.Linear(2048+memory_num, 1024+memory_num, bias = True) ### 64*64
        else:
            print('Please Confirm The Image Size')
        
        #self.fuse_att = nn.Linear(771, 259, bias = True) ### 64*64
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.dim_head = dim_head
        self.m = memory_num
        #self.m_k = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_k = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        #self.m_v = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.m_v = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        torch.nn.init.xavier_normal_(self.m_k)####不用这个初始化，损失函数是NAN
        torch.nn.init.xavier_normal_(self.m_v)
        
        self.Sm_k = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        #self.m_v = nn.Parameter(torch.FloatTensor(1, self.m, inner_dim))
        self.Sm_v = nn.Parameter(torch.empty(1, self.m, inner_dim),
                                     requires_grad=True)  # Tokenization parameters 
        torch.nn.init.xavier_normal_(self.Sm_k)####不用这个初始化，损失函数是NAN
        torch.nn.init.xavier_normal_(self.Sm_v)
        
        
    def forward(self, inp, x, y):
        b, n, _, h = *inp.shape, self.heads
        #context = default(context, inp)

        #if kv_include_self:
            #context = torch.cat((x, context), dim = 1) # cross token attention requires CLS token includes itself as key / value
        
        context = torch.cat([x, y], 1)    
        qkv = (self.to_q(inp), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        
        nk = k.shape[2]### token数
        m_k = np.sqrt(self.dim_head) * self.m_k.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        m_v = np.sqrt(self.m) * self.m_v.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        #print(k.shape, v.shape, m_k.shape, m_v.shape)
        #q = q.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([k, m_k], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([v, m_v], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #print(111111111111, k.shape, v.shape)### 100 8 12 64#'''


        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        Cross_attn = self.attend(dots)
        #Cross_attn = self.dropout(Cross_attn)#4 16 257 774
        
        S_k,S_v = self.to_kv(inp).chunk(2, dim = -1)
        S_k = rearrange(S_k, 'b n (h d) -> b h n d', h = h)
        S_v = rearrange(S_v, 'b n (h d) -> b h n d', h = h)
        
        
        nk = S_k.shape[2]### token数
        Sm_k = np.sqrt(self.dim_head) * self.Sm_k.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        Sm_v = np.sqrt(self.m) * self.Sm_v.expand(b, self.m, self.heads*self.dim_head).view(b, self.heads, self.m, self.dim_head)
        #print(k.shape, v.shape, m_k.shape, m_v.shape)
        #q = q.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        
        Sk = torch.cat([S_k, Sm_k], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        Sv = torch.cat([S_v, Sm_v], 2).view(b, self.heads, nk + self.m, self.dim_head)#.permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        #print(111111111111, k.shape, v.shape)### 100 8 12 64'''
        
        dots = einsum('b h i d, b h j d -> b h i j', q, Sk) * self.scale
        Self_attn = self.attend(dots)
        #Self_attn = self.dropout(Self_attn)#4 16 257 260
        
        #print(11111111111, Cross_attn.shape, Self_attn.shape)
        Cross_attn = self.fuse_att(Cross_attn)
        attn = Cross_attn + Self_attn

        out = einsum('b h i j, b h j d -> b h i d', attn, Sv)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x
        
# cross token attention transformer

class Cross_Attention(nn.Module):
    def __init__(self, h_dim, s_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(h_dim, s_dim, LayerNormalize(s_dim, CMAttention(s_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(s_dim, h_dim, LayerNormalize(h_dim, CMAttention(h_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, h_tokens, l_tokens):
        (h_cls, h_patch_tokens), (l_cls, l_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (h_tokens, l_tokens))
        ###  分别取出两模态的分类token和块token信息
        for h_attend_lg, l_attend_h in self.layers:
            h_cls = h_attend_lg(h_cls, context = l_patch_tokens, kv_include_self = True) + h_cls
            l_cls = l_attend_h(l_cls, context = h_patch_tokens, kv_include_self = True) + l_cls

        h_tokens = torch.cat((h_cls, h_patch_tokens), dim = 1)
        l_tokens = torch.cat((l_cls, l_patch_tokens), dim = 1)
        return h_tokens, l_tokens
        
        
# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)