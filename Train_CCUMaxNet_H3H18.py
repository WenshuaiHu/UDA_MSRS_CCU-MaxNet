import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime
from networks.CCUMaxNet import CCUMaxNet
from networks.discriminator import FCDiscriminator, OutspaceDiscriminator, FCDiscriminator1, FCDiscriminator2, FCDiscriminator3, PixelDiscriminator, soft_label_cross_entropy
from utils.highDHA_utils import adjust_learning_rate, adjust_learning_rate_D
from utils.pyt_utils import compute_cm, compute_IoU, compute_mIoU, compute_OA, compute_f1, compute_kappa, plot, recover
from loss.criterion import CrossEntropy, DiceLoss
import numpy as np
import pandas as pd
import argparse
from data import dataset as dataset
import random
import os
import csv
import matplotlib.pyplot as plt
import shutil
from networks.prob_2_entropy import prob_2_entropy
from operator import truediv
from networks.CCUMaxNet_Generator import ResnetGenerator, define_D, GANLoss, backward_D_basic, ImagePool
import itertools
import time 
from thop import profile, clever_format
import copy
parser = argparse.ArgumentParser(description='Cross-City Semantic Segmentation')
parser.add_argument('--fix_random', default=True, help='fix randomness')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu_id', default='0', help='gpu id')
# a 6000 beijing 1000
parser.add_argument('--epoch', default=3000, type=int, help='number of epoch')
parser.add_argument('--loop_epoch', default=50, type=int, help='number of epochs per validate')

parser.add_argument('--model', default='CCU_MaxNet', choices=['CCU_MaxNet'], type=str)

# dataset parameters
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--patch', default=64, type=int, help='input data size')
parser.add_argument('--num_channels', default=128, type=int, help='input channel number')
parser.add_argument('--overlay', default=0.5, type=float, help='overlay size')
parser.add_argument('--dataset', choices=['augsburg_berlin', 'nashua_hanover', 'houston1318', 'houston13_trento', 'trento_muufl'], default='houston1318', type=str, help='dataset to use')
parser.add_argument('--pca_flag', default=False, type=bool, help='weather use PCA dimension reduction on dataset')  # 10
parser.add_argument('--pca_num', default=10, type=int, help='pca component number')  # 10

parser.add_argument('--band_norm_flag', default=True, help='normalization by band')
parser.add_argument('--backbone_flag', default=False, help='use_backbone_pretrain')
parser.add_argument('--aug_flag', default=True, help='use_augment')
parser.add_argument('--backbone_path', default='./pretrain/hrnetv2_w48_imagenet_pretrained.pth', help='use_backbone')
# loss
parser.add_argument('--add_dice', default=True, type=bool, help='loss func')
# optimizer parameters
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--learning_rate_D', default=5e-4, type=float)
parser.add_argument('--lr_decay', default=20, type=int)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--power', default=0.9, type=float)
args = parser.parse_args()

def main(num_result):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    a = torch.cuda.is_available()
    if torch.cuda.is_available():
        print('GPU is true')
        print('Cuda Version: {}'.format(torch.version.cuda))
    else:
        print('CPU is true')

    if args.fix_random:
        manualSeed = args.seed
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)

        cudnn.deterministic = True
        cudnn.benchmark = False

    else:
        cudnn.benchmark = True
    print('Using GPU: {}'.format(args.gpu_id))

    # create dataset and model
    results_file = "results_{}.txt".format(args.model+args.dataset)
    label_train_loader, label_valid_loader, label_test_loader, num_classes, band_hsi, band_lidar = dataset.getdata(
        args.dataset,
        args.patch,
        args.overlay,
        args.batch_size,
        args.pca_flag,
        args.pca_num,
        args.band_norm_flag,
        args.aug_flag
    )
    
    criterion_ce = CrossEntropy(ignore_label=0)
    # denominator is the square term
    criterion_dice = DiceLoss(ignore_label=0, smooth=1e-5, p=1)
    #criterion_mse = nn.MSELoss()
    criterion_mse = nn.BCEWithLogitsLoss()

    # create model
    model = CCUMaxNet(band_hsi, args.patch, num_classes, num_channels = args.num_channels).cuda()
    
    '''
    ### 循环一致性损失
    '''
    #model_t2s = CCUMaxNet(band_hsi, args.patch, num_classes, num_channels = args.num_channels).cuda()
    model_reconstruct_s2t = ResnetGenerator((copy.deepcopy(model)).Swin_MAE, 
                                        args.patch, input_nc=num_classes, 
                                        num_channels_input = args.num_channels, num_fusion = model.fuse_out, num_channels_MAE=model.MAE_dim, 
                                        band_hsi=band_hsi, band_lidar=band_lidar).cuda()
                                        
    model_reconstruct_t2s = ResnetGenerator(model.Swin_MAE, 
                                        args.patch, input_nc=num_classes, 
                                        num_channels_input = args.num_channels, num_fusion = model.fuse_out, num_channels_MAE=model.MAE_dim, 
                                        band_hsi=band_hsi, band_lidar=band_lidar).cuda()
    
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), 
                                model_reconstruct_s2t.parameters(),
                                #model_t2s.parameters(), 
                                model_reconstruct_t2s.parameters()), 
                                lr=args.learning_rate, weight_decay=args.weight_decay)
    
    netD_t_f = define_D(model.fuse_out,  128, gpu_ids = int(args.gpu_id)).cuda()
    netD_t_x = define_D(band_hsi,  128, gpu_ids = int(args.gpu_id)).cuda()
    netD_t_y = define_D(band_lidar,  128, gpu_ids = int(args.gpu_id)).cuda()
    
    criterionGAN = nn.MSELoss()
    criterionCycle = nn.L1Loss()
    
    optimizer_fxy = torch.optim.Adam(itertools.chain(netD_t_f.parameters(), netD_t_x.parameters(), netD_t_y.parameters()), 
                                      lr=args.learning_rate_D, betas=(0.9, 0.999))
    
    # Generated image pool
    num_pool = 10
    fake_f_pool = ImagePool(num_pool)
    fake_x_pool = ImagePool(num_pool)
    fake_y_pool = ImagePool(num_pool)
    
    lambda_x = 10
    lambda_y = 10
    # lambda_l = 10
    '''
    ### 循环一致性损失
    '''
    # load pretrained backbone
    # if args.backbone_flag:
        # saved_state_dict = torch.load(args.backbone_path)
        # model_dict = model.state_dict()
        # for k, v in list(saved_state_dict.items()):
            # if k == 'conv1.weight':
                # saved_state_dict.pop(k)
            # if k == 'conv1.bias':
                # saved_state_dict.pop(k)
            # if k == 'bn1.weight':
                # saved_state_dict.pop(k)
            # if k == 'bn1.bias':
                # saved_state_dict.pop(k)
            # if k == 'bn1.running_mean':
                # saved_state_dict.pop(k)
            # if k == 'bn1.running_var':
                # saved_state_dict.pop(k)
            # if k == 'bn1.num_batches_tracked':
                # saved_state_dict.pop(k)
            # if k == 'conv2.weight':
                # saved_state_dict.pop(k)
            # if k == 'conv2.bias':
                # saved_state_dict.pop(k)
            # if k == 'bn2.weight':
                # saved_state_dict.pop(k)
            # if k == 'bn2.bias':
                # saved_state_dict.pop(k)
            # if k == 'bn2.running_mean':
                # saved_state_dict.pop(k)
            # if k == 'bn2.running_var':
                # saved_state_dict.pop(k)
            # if k == 'bn2.num_batches_tracked':
                # saved_state_dict.pop(k)
            # if str.find(k, 'last_layer') != -1:
                # saved_state_dict.pop(k)
        # saved_state_dict = {k: v for k, v in saved_state_dict.items()
                            # if k in model_dict.keys()}
        # model_dict.update(saved_state_dict)
        # misskey, _ = model.load_state_dict(model_dict, strict=False)
        # layer1 = model.layer1.state_dict()
        # model.msi_layer1.load_state_dict(layer1)
        # model.sar_layer1.load_state_dict(layer1)
        # print("load pretrained backbone")

    # label for adversarial training
    # source_label = 0
    # target_label = 1
    save_root_path = os.path.join(os.path.abspath(''),
                                  'result', args.model,
                                  args.dataset, "add_dice" if args.add_dice else "no_dice",
                                  "add_pca" if args.pca_flag else "no_pca",
                                  "band_norm" if args.band_norm_flag else "all_norm",
                                  "bz_" + str(args.batch_size),
                                  "use_pretrain" if args.backbone_flag else "no_pretrain",
                                  'num_result'+str(num_result),
                                  "aug" if args.aug_flag else "no_aug"
                                  )

    if os.path.exists(save_root_path):
        shutil.rmtree(save_root_path)
    miou_score = []
    max_miou = 0
    
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
        
    '''
    input = torch.randn(1, band_hsi, args.patch, args.patch).cuda()
    input_z = torch.randn(1, 1, args.patch, args.patch).cuda()
    flops, params = profile(model, inputs=(input, input_z, 'source'))
    #flops, params = clever_format([flops, params], '%.6f') 
    ## flops：26075178 / 1000**2  单位是M（兆），26075178 / 1000**3 单位是G。
    ## params：258194 / 1000 单位是K（千），

    input_s2tx = torch.randn(1, args.patch, 256).cuda()
    input_s2ty = torch.randn(1, args.patch, 256).cuda()
    input_s2tM1 = torch.randn(1, args.patch, 128).cuda()
    input_s2tyM2 = torch.randn(1, args.patch, 256).cuda()
    input_s2tt1 = torch.randn(1, 64, args.patch//2, args.patch//2).cuda()
    input_s2tyt2 = torch.randn(1, 64, args.patch//2, args.patch//2).cuda()
    
    M_curr_lists2t = []
    M_curr_lists2t.append(input_s2tM1)
    M_curr_lists2t.append(input_s2tyM2)
    xytemp1s2t = []
    xytemp1s2t.append(input_s2tt1)
    xytemp1s2t.append(input_s2tyt2)
    
    flops_DAs2t, params_DAs2t = profile(model_reconstruct_s2t, inputs=(input_s2tx, input_s2ty, M_curr_lists2t, xytemp1s2t))
    #flops_DAs2t, params_DAs2t = clever_format([flops_DAs2t, params_DAs2t], '%.6f') 

    input_t2sx = torch.randn(1, args.patch, 256).cuda()
    input_t2sy = torch.randn(1, args.patch, 256).cuda()
    input_t2sM1 = torch.randn(1, args.patch, 128).cuda()
    input_t2syM2 = torch.randn(1, args.patch, 256).cuda()
    input_t2st1 = torch.randn(1, 64, args.patch//2, args.patch//2).cuda()
    input_t2syt2 = torch.randn(1, 64, args.patch//2, args.patch//2).cuda()
    
    M_curr_listt2s = []
    M_curr_listt2s.append(input_s2tM1)
    M_curr_listt2s.append(input_s2tyM2)
    xytemp1t2s = []
    xytemp1t2s.append(input_s2tt1)
    xytemp1t2s.append(input_s2tyt2)
    
    flops_DAt2s, params_DAt2s = profile(model_reconstruct_t2s, inputs=(input_t2sx, input_t2sy, M_curr_listt2s, xytemp1t2s))
    #flops_DAt2s, params_DAt2s = clever_format([flops_DAt2s, params_DAt2s], '%.6f') 

    input_f = torch.randn(1, 192, args.patch//2, args.patch//2).cuda()
    flops_DAf, params_DAf = profile(netD_t_f, inputs=(input_f))
    #flops_DAf, params_DAf = clever_format([flops_DAf, params_DAf], '%.6f') 
    

    input_x = torch.randn(1, band_hsi, args.patch, args.patch).cuda()
    flops_DAx, params_DAx = profile(netD_t_x, inputs=(input_x))
    #flops_DAx, params_DAx = clever_format([flops_DAx, params_DAx], '%.6f') 
    
    input_y = torch.randn(1, band_lidar, args.patch, args.patch).cuda()
    flops_DAy, params_DAy = profile(netD_t_y, inputs=(input_y))
    #flops_DAy, params_DAy = clever_format([flops_DAy, params_DAy], '%.6f') 
    
    #flops = flops + '-' +  flops_DAf + '-' +  flops_DAx + '-' +  flops_DAy + '-' +  flops_DAs2t + '-' +  flops_DAt2s
    flops = flops + flops_DAf + flops_DAx + flops_DAy + flops_DAs2t + flops_DAt2s
    params = params + params_DAf + params_DAx + params_DAy + params_DAs2t + params_DAt2s
    
    flops, params = clever_format([flops, params], '%.6f')
    
    print("1111111111111-flops:", flops)
    print("1111111111111-params:", params)
    #print("1111111111111-requires_grad_params:", count_parameters(feature_encoder)/1e6)
    flops_params_writer = save_root_path + '/flops_params.txt'
        
    with open (flops_params_writer, 'a') as  f_timecal:         
        f_timecal.write(str(flops) +'\n')
        f_timecal.write(str(params) +'\n')
    f_timecal.close()#'''
        
        
    for epoch in range(args.epoch):
        train_time_epoch = time.time()
        
        model.train()
        model_reconstruct_s2t.train()
        #model_t2s.train()
        model_reconstruct_t2s.train()
        netD_t_f.train()
        netD_t_x.train()
        netD_t_y.train()
        
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, args.learning_rate, epoch, args.epoch, args.power)

        optimizer_fxy.zero_grad()
        adjust_learning_rate_D(optimizer_fxy, args.learning_rate_D, epoch, args.epoch, args.power)

        # train G
        # don't accumulate grads in D            
        for param in netD_t_f.parameters():
            param.requires_grad = False
        for param in netD_t_x.parameters():
            param.requires_grad = False 
        for param in netD_t_y.parameters():
            param.requires_grad = False
            
        # train with source
        try:
            _, traindata = next(enumerate(label_train_loader))
        except StopIteration:
            trainloader_iter = iter(label_train_loader)
            _, traindata = next(trainloader_iter)

        if args.dataset == 'augsburg' or args.dataset == 'beijing':
            x, y, z, label = traindata
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            label = label.cuda()

            feat_source, pred_source, loss_rec = model(x, y, z, 'source')
        elif args.dataset == 'nashua_hanover' or args.dataset == 'houston1318' or args.dataset ==  'houston13_trento' or args.dataset == 'trento_muufl' or 'augsburg_berlin':
            x, y, label, label_GAN = traindata
            x = x.cuda()
            y = y.cuda()
            label = label.cuda()
            label_GAN = label_GAN.cuda()
            
            feat_source, feat_x_source, feat_y_source, pred_source, loss_rec, M_curr_list, xytemp1 = model(x, y, 'source')
        else:
            raise ValueError("Unknown dataset")
        x_real_source = x
        y_real_source = y
            
        '''
        ### 循环一致性损失
        '''
        loss = criterion_ce(pred_source, label)
        loss = loss + loss_rec
        # if args.add_dice:
        #     loss += criterion_dice(pred_source, label)

        # train with cycleGAN
        try:
            _, validdata = next(enumerate(label_valid_loader))
        except StopIteration:
            targetloader_iter = iter(label_valid_loader)
            _, validdata = next(targetloader_iter)
            
        if args.dataset == 'augsburg' or args.dataset == 'beijing':
            x, y, z, label_val = validdata
            x = x.cuda()
            y = y.cuda()
            z = z.cuda()
            # label = label.cuda()
            feat_target, pred_target, loss_rec = model(x, y, z, 'target')
            
        elif args.dataset == 'nashua_hanover' or args.dataset == 'houston1318' or args.dataset ==  'houston13_trento' or args.dataset == 'trento_muufl' or 'augsburg_berlin':
            x, y, _ = validdata
            x = x.cuda()
            y = y.cuda()
            # label = label.cuda()
            feat_target, feat_x_target, feat_y_target, _, loss_rec, M_curr_list_t, xytemp1_t = model(x, y, 'target')
            #_, _, _, _, loss_rec_, _, _ = model_t2s(x, y, 'target')
        else:
            raise ValueError("Unknown dataset")
        x_real_target = x
        y_real_target = y
        
        '''
        ### 循环一致性损失
        '''
        ## source 2 target 将源域数据转换为目标域的数据风格（但本身地物还是源域的），比如：传感器特性、光照等风格
        feat_source_real = feat_source
        
        #print(1111111111111, feat_x_source.shape, feat_y_source.shape, M_curr_list_t[0].shape, M_curr_list_t[1].shape, xytemp1_t[0].shape, xytemp1_t[1].shape)
        
        x_rec_target_fake, y_rec_target_fake = model_reconstruct_s2t(feat_x_source, feat_y_source, M_curr_list_t, xytemp1_t) ## G_B
        
        ## target 2 source  循环一致性（其实是两种风格之间的一致性（不同源域））
        feat_source_fake, feat_x_source_fake, feat_y_source_fake, pred_source_from_target, loss_rec_target_from_source, _, _ = model(x_rec_target_fake, y_rec_target_fake, 'source') ## G_A
        x_rec_source_fake, y_rec_source_fake = model_reconstruct_t2s(feat_x_source_fake, feat_y_source_fake, M_curr_list, xytemp1) ## G_B
        _, _, _, pred_source_from_target_, loss_rec_target_from_source_, _, _ = model(x_rec_source_fake, y_rec_source_fake, 'source') ## G_A

        #源域x和y的循环一致性
        x_real_source = x_real_source
        y_real_source = y_real_source
        x_rec_source_fake = x_rec_source_fake
        y_rec_source_fake = y_rec_source_fake

        loss_cycle_x = criterionCycle(x_rec_source_fake, x_real_source) * lambda_x
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_y = criterionCycle(y_rec_source_fake, y_real_source) * lambda_y
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_source = (loss_cycle_x + loss_cycle_y)*0.5

        #源域 语义一致性，其实是两种不同风格的源域数据，因为地物信息是相同的，所以 标签应该也一致。
        #loss_pred_target_from_source = criterion_ce(pred_source_from_target, label) + loss_rec_target_from_source
        loss_pred_target_from_source_ = criterion_ce(pred_source_from_target_, label)# + loss_rec_target_from_source_
        
        '''
        ### 循环一致性损失
        '''
        
        loss_pred_source_from_source = loss
        
        loss_segmentation = loss_pred_source_from_source + loss_pred_target_from_source_*0.5 + loss_rec*0.5 + loss_cycle_source
        #loss_segmentation.backward()    
  
        '''
        ### adv损失
        '''
        x_real_target = x_real_target
        y_real_target = y_real_target
        x_rec_target_fake = x_rec_target_fake#.detach()
        y_rec_target_fake = y_rec_target_fake#.detach()
        
        #feat_target, _, _, _, _, _, _ = model(x, y, 'target')
        #feat_target_fake, _, _, _, _, _, _ = model(x_rec_target_fake, y_rec_target_fake, 'target')
        
        feat_target_real = feat_target
        feat_target_fake = feat_source_fake
        
        #目标域 特征级 GAN 损失
        feat_target_fake_dis = netD_t_f(feat_target_fake)
        loss_G_t_f = criterionGAN(feat_target_fake_dis, torch.ones_like(feat_target_fake_dis))

        #目标域 像素级 GAN损失
        x_rec_target_fake_dis = netD_t_x(x_rec_target_fake)
        loss_G_t_x = criterionGAN(x_rec_target_fake_dis, torch.ones_like(x_rec_target_fake_dis))
        
        y_rec_target_fake_dis = netD_t_y(y_rec_target_fake)
        loss_G_t_y = criterionGAN(y_rec_target_fake_dis, torch.ones_like(y_rec_target_fake_dis))
        
        loss_cycle = loss_segmentation +  (loss_G_t_f + loss_G_t_x + loss_G_t_y)*0.5#'''

        
        loss_cycle.backward()        
        
        '''
        ### adv损失
        '''
        optimizer.step()
        

        '''
        ### 像素级和特征级对抗损失
        '''
        for param in netD_t_f.parameters():
            param.requires_grad = True
        for param in netD_t_x.parameters():
            param.requires_grad = True 
        for param in netD_t_y.parameters():
            param.requires_grad = True
            
        x_real_target = x_real_target
        x_rec_target_fake = x_rec_target_fake
        
        y_real_target = y_real_target
        y_rec_target_fake = y_rec_target_fake
        

        feat_target_real = feat_target
        feat_target_fake = feat_target_fake
        
        '''
        # feat_target_fake = fake_f_pool.query(feat_target_fake)## 记录旧的生成的图像，用于增加鉴别器稳定性 参考 https://zhuanlan.zhihu.com/p/519967273
        ### 源域
        feat_target_real = netD_t_f(feat_target_real.detach())
        loss_D_f_real = criterionGAN(feat_target_real, torch.ones_like(feat_target_real))
        
        x_real_target = netD_t_x(x_real_target.detach())
        loss_D_x_real = criterionGAN(x_real_target, torch.ones_like(x_real_target))
        
        y_real_target = netD_t_y(y_real_target.detach())
        loss_D_y_real = criterionGAN(y_real_target, torch.ones_like(y_real_target))
        
        loss_D_source = (loss_D_f_real + loss_D_x_real + loss_D_y_real)*0.5
        loss_D_source.backward()
        
        ### 目标域
        feat_target_fake = netD_t_f(feat_target_fake.detach())
        loss_D_f_fake= criterionGAN(feat_target_fake, torch.zeros_like(feat_target_fake))
        
        x_rec_target_fake = netD_t_x(x_rec_target_fake.detach())
        loss_D_x_fake = criterionGAN(x_rec_target_fake, torch.zeros_like(x_rec_target_fake))
        
        y_rec_target_fake = netD_t_y(y_rec_target_fake.detach())
        loss_D_y_fake = criterionGAN(y_rec_target_fake, torch.zeros_like(y_rec_target_fake))
        
        loss_D_target = (loss_D_f_fake + loss_D_x_fake + loss_D_y_fake)*0.5
        loss_D_target.backward()#'''
        
        ### 源域 + 目标域
        feat_target_fake = fake_f_pool.query(feat_target_fake)## 
        loss_D_f_real, loss_D_f_fake = backward_D_basic(netD_t_f, feat_target_real.detach(), feat_target_fake.detach(), criterionGAN)
        loss_D_f = (loss_D_f_real + loss_D_f_fake)*0.5
        loss_D_f.backward()
        
        x_rec_target_fake = fake_x_pool.query(x_rec_target_fake)
        loss_D_x_real, loss_D_x_fake = backward_D_basic(netD_t_x, x_real_target.detach(), x_rec_target_fake.detach(), criterionGAN)
        loss_D_x = (loss_D_x_real + loss_D_x_fake)*0.5
        loss_D_x.backward()
        
        y_rec_target_fake = fake_y_pool.query(y_rec_target_fake)
        loss_D_y_real, loss_D_y_fake = backward_D_basic(netD_t_y, y_real_target.detach(), y_rec_target_fake.detach(), criterionGAN)
        loss_D_y = (loss_D_y_real + loss_D_y_fake)*0.5
        loss_D_y.backward()
        
        optimizer_fxy.step()
        
        '''
        ### 像素级和特征级对抗损失
        '''
        
        #print("Epoch: {:03d}, loss_seg: {:.4f}, loss_cycle: {:.4f}, loss_D_source: {:.4f}, loss_D_target: {:.4f}" .format(epoch + 1, loss_segmentation, loss_cycle, loss_D_source, loss_D_target))
        print("Epoch: {:03d}, loss_seg: {:.4f}, loss_cycle: {:.4f}, loss_D_f: {:.4f}, loss_D_x: {:.4f}, loss_D_y: {:.4f}" .format(epoch + 1, loss_segmentation, loss_cycle, loss_D_f, loss_D_x, loss_D_y))
        
        train_epoch_duration = time.time() - train_time_epoch
        time_writer = save_root_path +'\\time_cal_{}.txt'.format(num_result)
        with open (time_writer, 'a') as  f_timecal:         
            f_timecal.write(str(train_epoch_duration)+'\n')##y_true.cpu().numpy
            f_timecal.close()

        if (epoch + 1) % args.loop_epoch == 0:
            model.eval()
            model_reconstruct_s2t.eval()
            #model_t2s.eval()
            model_reconstruct_t2s.eval()
            netD_t_f.eval()
            netD_t_x.eval()
            netD_t_y.eval()



            label_total = []
            label_gt = []
            start_time = time.time()
            for i, testdata in enumerate(label_test_loader):
                if args.dataset == 'augsburg' or args.dataset == 'beijing':
                    x, y, z, label_val = testdata
                    x = x.cuda()
                    y = y.cuda()
                    z = z.cuda()
                    label = label.cuda()

                    _, output_label, _ = model(x, y, z, 'target')
                elif args.dataset == 'nashua_hanover' or args.dataset == 'houston1318' or args.dataset ==  'houston13_trento' or args.dataset == 'trento_muufl' or 'augsburg_berlin':
                    x, y, label = testdata
                    x = x.cuda()
                    y = y.cuda()
                    label = label.cuda()
                    _, _, _, output_label, _, _, _ = model(x, y, 'target')
                else:
                    raise ValueError("Unknown dataset")                

                pred = output_label.cpu().detach().numpy().transpose(0, 2, 3, 1)
                seg_pred = np.asarray(np.argmax(pred[:, :, :, 1:], axis=3), dtype=np.uint8)
                label_total.append(seg_pred + 1)

                label = label.cpu().detach().numpy()
                label_gt.append(label)
                # print("valid epoch: {:03d}".format(i))
            duration = time.time() - start_time

            save_path = os.path.join(save_root_path, "epoch_" + str(epoch + 1))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pd.DataFrame(np.array(label_gt).flatten()).to_csv(save_path + '/label_gt.csv', index=False)
            pd.DataFrame(np.array(label_total).flatten()).to_csv(save_path + '/label_pre.csv', index=False)
            label_gt, label_total = recover(label_gt, label_total, args.dataset, args.patch, save_path)
            
            cm = compute_cm(label_gt, label_total)
            pd.DataFrame(cm).to_csv(save_path + '/confuse_matrix.csv', index=True)

            # IoU = compute_IoU(cm)
            # mIoU = compute_mIoU(cm)
            f1 = compute_f1(label_total, label_gt)
            # oa = compute_OA(cm)
            # kappa = compute_kappa(label_total, label_gt)
            
            oa = 1. * np.trace(cm) / np.sum(cm)
            # compute Producer Accuracy (PA)
            #IoU = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(num_classes-1)])

            list_diag = np.diag(cm)
            list_raw_sum = np.sum(cm, axis=1)
            IoU = np.nan_to_num(truediv(list_diag, list_raw_sum))            
            
            # compute Average Accuracy (AA)
            mIoU = np.mean(IoU)
            
            # compute kappa coefficient
            pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
            kappa = (oa - pe) / (1 - pe)

            IoU_str = [f'{item:.4f}' for item in IoU]

            if mIoU > max_miou:
                max_miou = mIoU
                max_miou_epoch = epoch
                plot(label_gt, label_total, save_path)
                print("Epoch: {:03d}, max_miou_Epoch: {:03d}, max_miou: {:.4f}".format(epoch + 1, max_miou_epoch + 1, max_miou))
                print("save best weight:{:03d}".format(epoch + 1))
                ### 出现中文路径，保存模型会报错
                torch.save(model.state_dict(), os.path.join(save_root_path, 'best_model.pth'))
                torch.save(model_reconstruct_s2t.state_dict(), os.path.join(save_root_path, 'best_cycleG_B.pth'))
                #torch.save(model_t2s.state_dict(), os.path.join(save_root_path, 'best_model_t.pth'))
                torch.save(model_reconstruct_t2s.state_dict(), os.path.join(save_root_path, 'best_cycleG_B_t.pth'))
                torch.save(netD_t_f.state_dict(), os.path.join(save_root_path, 'best_netD_l.pth'))
                torch.save(netD_t_x.state_dict(), os.path.join(save_root_path, 'best_netD_x.pth'))
                torch.save(netD_t_y.state_dict(), os.path.join(save_root_path, 'best_netD_y.pth'))
                
                
                with open(results_file, "a") as f:
                    # 记录每个epoch对应的train_loss、lr以及验证集各指标
                    info = f"[epoch: {max_miou_epoch + 1}]\n" \
                           f"max_miou: {mIoU:.6f}\n"
                    f.write(info + "\n\n")
            with open(save_path + "/eval_index_OA{}_mIoU{}.csv".format(str(round(oa, 3)), str(round(mIoU, 3))), 'w', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["IoU:"])
                csv_writer.writerow([IoU_str])
                csv_writer.writerow(["mIoU: {:.4f}".format(mIoU)])
                csv_writer.writerow(["f1: {:.4f}".format(f1)])
                csv_writer.writerow(["OA: {:.4f}".format(oa)])
                csv_writer.writerow(["kappa: {:.4f}".format(kappa)])
                csv_writer.writerow(["Test Time (s): {:.6f}".format(duration)])
            miou_score.append(mIoU)

    pd.DataFrame(miou_score).to_csv(os.path.join(save_root_path, 'miou_score.csv'), index=False, header=None)

    x = list(np.arange(0, args.epoch, args.loop_epoch) + args.loop_epoch)
    fig = plt.figure()
    plt.plot(x, miou_score)
    fig.savefig(os.path.join(save_root_path, 'miou_score.jpg'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    for num_result in range(8):
        main(num_result)
