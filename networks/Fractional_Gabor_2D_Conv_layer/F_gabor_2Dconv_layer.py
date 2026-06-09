import torch
import torch.nn as nn
from ..Fractional_Gabor_2D_Conv_layer.opsfrac import gen_gf_list
from torch.nn.parameter import Parameter
import numpy as np 

def F_gabor_conv(input_, f, out_channels, kernel_size=3, order = 25, bias=None, stride=1, padding=1, dilation=1, groups=1):
    '''
    input_: 输出张量数据
    out：输出通道数，实际输出通道数为4*out_channel
    order: fractional order
    '''
    inp = input_.shape[1]
    filter_size = [kernel_size, kernel_size, out_channels, inp]
    gf_1 = gen_gf_list(filter_size, f, order)
    #gf_2 = gen_gf_list(filter_size_1, f, order)
    #gf_3 = gen_gf_list(filter_size_1, f, order)

    filter_base = torch.randn(kernel_size, kernel_size, out_channels, inp).cuda()
    
    filter_0 = Parameter(data=filter_base*torch.from_numpy(gf_1[0]).cuda()).reshape(out_channels, inp, kernel_size, kernel_size).type(torch.FloatTensor).cuda()
    filter_1 = Parameter(data=filter_base*torch.from_numpy(gf_1[1]).cuda()).reshape(out_channels, inp, kernel_size, kernel_size).type(torch.FloatTensor).cuda()
    filter_2 = Parameter(data=filter_base*torch.from_numpy(gf_1[2]).cuda()).reshape(out_channels, inp, kernel_size, kernel_size).type(torch.FloatTensor).cuda()
    filter_3 = Parameter(data=filter_base*torch.from_numpy(gf_1[3]).cuda()).reshape(out_channels, inp, kernel_size, kernel_size).type(torch.FloatTensor).cuda()
    
    F_gabor_conv0 = nn.functional.conv2d(input_, filter_0, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups) 
    F_gabor_conv1 = nn.functional.conv2d(input_, filter_1, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups) 
    F_gabor_conv2 = nn.functional.conv2d(input_, filter_2, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups) 
    F_gabor_conv3 = nn.functional.conv2d(input_, filter_3, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups) 
    
    F_gabor_conv = [F_gabor_conv0]
    F_gabor_conv.append(F_gabor_conv1)
    F_gabor_conv.append(F_gabor_conv2)
    F_gabor_conv.append(F_gabor_conv3)
    F_gabor_conv = torch.cat(F_gabor_conv, dim=1)#   
    
    
    return F_gabor_conv
'''
if __name__ == '__main__':
    order=25
    f = [1., 1./2, 1./3, 1./4]
    inputs = torch.randn(20, 16, 11, 11).type(torch.DoubleTensor)
    output = F_gabor_conv(inputs, f, 8, kernel_size=3, order = order, stride=1, padding=1, dilation=1, groups=1)
    output = nn.functional.hardswish(output, inplace=False)
    print(11111111, inputs.size(), output.size())### '''