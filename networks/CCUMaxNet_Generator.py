# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:27:26 2024

@author: user
"""
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import random
from einops import rearrange
import torch.nn.functional as F

BN_MOMENTUM = 0.01

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        if len(input.shape) < 4:
            input = input.unsqueeze(0)
        return self.net(input)

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     net.to(gpu_ids[0])
    #     net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
        
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, Swin_MAE, image_size, input_nc=7, num_fusion = 192, num_channels_MAE=64, 
                 num_channels_input = 128, band_hsi=48, band_lidar=1, ngf=64, patch_size = 2, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # if type(norm_layer) == functools.partial:
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
        #          norm_layer(ngf),
        #          nn.ReLU(True)]

        # n_downsampling = 2
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #               norm_layer(ngf * mult * 2),
        #               nn.ReLU(True)]

        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):       # add ResNet blocks

        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, num_channels, kernel_size=7, padding=0)]
        # model += [nn.Tanh()]

        # self.model = nn.Sequential(*model)
        # self.num_channelsx = num_channels//2
        # self.num_channelsy = num_channels//2
        '''
        self.transconvx = nn.Sequential(
            # nn.Conv2d(self.fuse_out, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(self.fuse_out, 512, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            nn.Conv2d(num_fusion, num_channels_MAE*4, kernel_size=6, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(num_channels_MAE*4, momentum=BN_MOMENTUM),
            #nn.ReLU(inplace=False),
            hswish()
            # nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
        )
        self.transconvy = nn.Sequential(
            # nn.Conv2d(self.fuse_out, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(self.fuse_out, 512, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            nn.Conv2d(num_fusion, num_channels_MAE*4, kernel_size=6, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(num_channels_MAE*4, momentum=BN_MOMENTUM),
            #nn.ReLU(inplace=False),
            hswish()
            # nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
            # nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            # nn.ReLU(inplace=False),
        )'''
        #self.final_conv = nn.Conv2d(input_nc, num_fusion, 1, 1)
        
        self.dncoder = Swin_MAE.Dncoder
        self.image_size = image_size
        
        ###HSI
        self.conv1 = nn.Conv2d(num_channels_input, band_hsi, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(band_hsi, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(band_hsi, band_hsi, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(band_hsi, momentum=BN_MOMENTUM)
        #self.relu = nn.ReLU(inplace=False)
        self.relu = hswish()
        
        ###LiDAR
        self.conv_lidar = nn.Conv2d(num_channels_input, band_lidar, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_msi = nn.BatchNorm2d(band_lidar, momentum=BN_MOMENTUM)
        self.num_channels_input = num_channels_input
        self.patch_size = patch_size
        
    def forward(self, input_x, input_y, M_curr_list, xytemp1):
        """Standard forward"""
        # print(11111111111111111, input.shape)
        # f = self.model(input)
        # f = self.final_conv(input)
        #f = input
        #x = self.transconvx(f)
        #y = self.transconvy(f)
        # print(11111111111111111111, f.shape, x.shape, y.shape)
        #x = rearrange(x, 'B C H W -> B (H W) C')
        #y = rearrange(y, 'B C H W -> B (H W) C')
        # print(11111111111111111111, x.shape, y.shape)
        x, y = self.dncoder(input_x, input_y, M_curr_list, xytemp1)
        x = rearrange(x, 'B (H W) (C d1 d2)-> B C (H d1) (W d2)', H = self.image_size//self.patch_size, W=self.image_size//self.patch_size, d1 =self.patch_size, d2=self.patch_size)
        y = rearrange(y, 'B (H W) (C d1 d2)-> B C (H d1) (W d2)', H = self.image_size//self.patch_size, W=self.image_size//self.patch_size, d1 =self.patch_size, d2=self.patch_size)
        # print(11111111111111111111, x.shape, y.shape)
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2( self.conv2(x)))
        
        y = self.relu(self.bn_msi(self.conv_lidar(y)))
        # print(11111111111111111111, x.shape, y.shape)
        return x, y
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    
    return init_net(net, init_type, init_gain, gpu_ids)
    
class ImagePool():
    
    """Image Pool https://zhuanlan.zhihu.com/p/519967273
        建立一个缓冲区，来存储之前50次训练生成的fake图像，然后从中以一般概率从缓冲区中随机取出之前存储的fake图像，
        反馈给鉴别器，彼此达到减少模型震荡的问题。    
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
    
    
##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - 任意.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))     # 将标签值保存到model中
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        self.loss = nn.MSELoss()        # 暂时将G_A与D_B的损失先不变

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

def backward_D_basic(netD, real, fake, criterionGAN):
    """Calculate GAN loss for the discriminator

    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator

    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    pred_real = netD(real)
    # loss_D_real = criterionGAN(pred_real, True)
    loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
    
    # Fake
    pred_fake = netD(fake.detach())
    # loss_D_fake = criterionGAN(pred_fake, False)
    loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
    
    # Combined loss and calculate gradients
    #loss_D = (loss_D_real + loss_D_fake)*0.5
    # loss_D.backward()
    return loss_D_real, loss_D_fake

    