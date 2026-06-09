import torch
import torch.nn as nn
import torch.utils.data as Data
import slidingwindow as sw
import numpy as np
import scipy.io as sio
import os
import h5py
from sklearn.decomposition import PCA
from imgaug import augmenters as iaa


def getdata(dataset, patch, overlay, batchsize, pac_flag=False, pca_num = 10, 
            band_norm_flag=False, aug_flag=False):
    label_train, label_valid, label_test, num_classes, band, band_lidar = slide_crop(dataset, patch, overlay, pac_flag, pca_num, band_norm_flag, aug_flag)
    label_test_loader = Data.DataLoader(label_test, batch_size=1, shuffle=False, drop_last=False)
    label_train_loader = Data.DataLoader(label_train, batch_size=batchsize, shuffle=True, num_workers=0,
                                         pin_memory=True, drop_last=True)
    label_valid_loader = Data.DataLoader(label_test, batch_size=batchsize, shuffle=True, num_workers=0,
                                         pin_memory=True, drop_last=True)
                                         
    return label_train_loader, label_valid_loader, label_test_loader, num_classes, band, band_lidar


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def band_normalization(data):
    """ normalize the matrix to (0,1), r.s.t A axis (Default=0)
        return normalized matrix and a record matrix for normalize back
    """
    size = data.shape
    if len(size) != 3:
        raise ValueError("Unknown dataset")
    for i in range(size[-1]):
        _range = np.max(data[:, :, i]) - np.min(data[:, :, i])
        data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / _range
    return data


def read_data(dataset, pca_flag=False, pca_num = 10, band_norm=False):
    if dataset == 'augsburg_berlin':
        num_classes, band = 14, 242
        train_file = r'data/Augsburg_Berlin/augsburg_multimodal.mat'
        col_train, row_train = 1360, 886
        valid_file = r'data/Augsburg_Berlin/berlin_multimodal.mat'
        col_valid, row_valid = 811, 2465
        input_data = sio.loadmat(train_file)
        valid_data = sio.loadmat(valid_file)

        hsi = input_data['HSI']  # 886 1360 242
        hsi = hsi.astype(np.float32)
        #msi = input_data['MSI']
        #msi = msi.astype(np.float32)
        lidar = input_data['SAR']#[:, :, 0]
        #a = -lidar
        #lidar = np.where(a == -0, 0, a)
        #lidar = lidar[:, :, np.newaxis]
        #lidar = normalization(lidar)
        lidar = lidar.astype(np.float32)
        label = input_data['label']

        hsi_valid = valid_data['HSI'][:, :, 0:band]  # 2456 811 242
        hsi_valid = hsi_valid.astype(np.float32)
        #msi_valid = valid_data['MSI']
        #msi_valid = msi_valid.astype(np.float32)
        lidar_valid = valid_data['SAR']#[:, :, 0]
        #b = -lidar_valid
        #lidar_valid = np.where(b == -0, 0, b)
        #lidar_valid = normalization(lidar_valid)
        #lidar_valid = lidar_valid[:, :, np.newaxis]
        lidar_valid = lidar_valid.astype(np.float32)
        label_valid = valid_data['label']

        # a = np.min(label_valid)
        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = pca_num

            del hsi_matrix

    elif dataset == 'beijing':
        num_classes = 14
        band = 116  # 116
        ## beijing is training, wuhan is testing
        train_file = r'data/data2/beijing.mat'
        train_file_label = r'data/data2/beijing_label.mat'
        col_train, row_train = 13474, 8706
        valid_file = r'data/data2/wuhan.mat'
        valid_file_label = r'data/data2/wuhan_label.mat'
        col_valid, row_valid = 6225, 8670

        with h5py.File(train_file, 'r') as f:
            f = h5py.File(train_file, 'r')
        hsi = np.array(f['HSI'])
        msi = np.transpose(f['MSI'])
        sar = np.transpose(f['SAR'])
        # idx = np.where(np.isnan(sar))
        sar[1097, 8105, 1] = sum([sar[1096, 8105, 1], sar[1098, 8105, 1], sar[1097, 8104, 1], sar[1097, 8106, 1]]) / 4

        # cut beijing suburb region
        cut_length = 0
        col_train = col_train - cut_length
        hsi = hsi[:, :, cut_length // 3:]
        msi = msi[cut_length:, :, :]
        sar = sar[cut_length:, :, :]

        # PCA
        if pca_flag:
            # applying PCA for HSI # hsi (116, 2903, 4492)
            hsi_matrix = np.reshape(np.transpose(hsi), (hsi.shape[1] * hsi.shape[2], hsi.shape[0]))  # 2903*4492 116
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 116*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2903*4492 10
            hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi.shape[2], hsi.shape[1], pca.n_components_)))
            
            band = pca_num
            del hsi
        else:
            hsi_cube = hsi

        mm = nn.Upsample(scale_factor=3, mode='nearest', align_corners=None)
        # upsample from 30m to 10m
        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi1 = np.transpose(hsi1)
        # remove extra pixels
        hsi = hsi1[:col_train, :row_train, :]
        del hsi1

        with h5py.File(train_file_label, 'r') as f:
            f = h5py.File(train_file_label, 'r')
        label = np.transpose(f['label'])
        # cut beijing label
        label = label[cut_length:, :]

        with h5py.File(valid_file, 'r') as f:
            f = h5py.File(valid_file, 'r')
        hsi_valid = np.array(f['HSI'])
        msi_valid = np.transpose(f['MSI'])
        sar_valid = np.transpose(f['SAR'])

        # PCA
        if pca_flag:
            ## applying PCA for valid HSI
            hsi_matrix = np.reshape(np.transpose(hsi_valid), (hsi_valid.shape[1] * hsi_valid.shape[2], hsi_valid.shape[0]))
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()
            hsi_matrix = np.matmul(hsi_matrix, newspace)
            hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi_valid.shape[2], hsi_valid.shape[1], pca.n_components_)))
            
            band = pca_num
            del hsi_valid
        else:
            hsi_cube = hsi
            
        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi_valid = np.transpose(hsi1)
        del hsi1

        with h5py.File(valid_file_label, 'r') as f:
            f = h5py.File(valid_file_label, 'r')
        label_valid = np.transpose(f['label'])

    elif dataset == 'nashua_hanover':
        num_classes = 8
        band = 114  # 116
        ## beijing is training, wuhan is testing
        hsi = r'data/Nashua_Hanover/Source/newNHSIdata.mat'
        hsi = sio.loadmat(hsi)['NHSInew']
        hsi = hsi.astype(np.float32) 
        lidar = r'data/Nashua_Hanover/Source/newNCHMdata.mat'
        lidar = sio.loadmat(lidar)['NDSMnew']
        lidar = lidar.astype(np.float32) 
        lidar = lidar[:, :, np.newaxis]
        label = r'data/Nashua_Hanover/Source/NGT7.mat'
        label = sio.loadmat(label)['NGT']
        col_train, row_train = 1050, 450
        
        hsi_valid = r'data/Nashua_Hanover/Target/HHSIdata.mat'
        hsi_valid = sio.loadmat(hsi_valid)['HHSInew1']
        hsi_valid = hsi_valid.astype(np.float32) 
        lidar_valid = r'data/Nashua_Hanover/Target/HCHMdata.mat'
        lidar_valid = sio.loadmat(lidar_valid)['HCHMnew1']
        lidar_valid = lidar_valid[:, :, np.newaxis]
        lidar_valid = lidar_valid.astype(np.float32) 
        label_valid = r'data/Nashua_Hanover/Target/HGT7.mat'
        label_valid = sio.loadmat(label_valid)['HGT']
        col_valid, row_valid = 620, 700

        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = pca_num

            del hsi_matrix
    elif dataset == 'houston1318':
        num_classes = 8
        band = 48  # 116
        ## beijing is training, wuhan is testing
        hsi = r'data/Houston13_18/Source/houston13all_hsi.mat'
        hsi = sio.loadmat(hsi)['houston13hsi']
        hsi = hsi.astype(np.float32) 
        lidar = r'data/Houston13_18/Source/lidar13.mat'
        lidar = sio.loadmat(lidar)['lidar13']
        lidar = lidar.astype(np.float32) 
        lidar = lidar[:, :, np.newaxis]
        label = r'data/Houston13_18/Source/houston13new.mat'
        label = sio.loadmat(label)['houston13new']
        col_train, row_train = 954, 210
        
        # print(111111111111111, label.shape, hsi.shape) ## (210, 954) (210, 954, 48)
        # zero_matrix = np.zeros((row_train, col_train, num_classes))
        # for i in range(num_classes):
        #     for j in range(row_train):
        #             for k in range(col_train):
        #                 if label[j, k] == i+1:
        #                     zero_matrix[j,k,:]=i+1
        # label = zero_matrix
        
        hsi_valid = r'data/Houston13_18/Target/houston18all_hsi.mat'
        hsi_valid = sio.loadmat(hsi_valid)['houston18hsi']
        hsi_valid = hsi_valid.astype(np.float32) 
        lidar_valid = r'data/Houston13_18/Target/lidar18.mat'
        lidar_valid = sio.loadmat(lidar_valid)['lidar18']
        lidar_valid = lidar_valid[:, :, np.newaxis]
        lidar_valid = lidar_valid.astype(np.float32) 
        label_valid = r'data/Houston13_18/Target/houston18new.mat'
        label_valid = sio.loadmat(label_valid)['houston18new']
        col_valid, row_valid = 954, 210

        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = pca_num

            del hsi_matrix
    elif dataset == 'houston13_trento':
        num_classes = 5
        band = 63  # 116
        ## beijing is training, wuhan is testing
        hsi = r'data/Houston13_Trento/Source/HtTsmall.mat'
        hsi = sio.loadmat(hsi)['houston13hsi']
        hsi = hsi.astype(np.float32) 
        lidar = r'data/Houston13_Trento/Source/lidar13.mat'
        lidar = sio.loadmat(lidar)['lidar13']
        lidar = lidar.astype(np.float32) 
        lidar = lidar[:, :, np.newaxis]
        label = r'data/Houston13_Trento/Source/HtTSmallnewGT.mat'
        label = sio.loadmat(label)['smallGT']
        col_train, row_train = 954, 210
        
        hsi_valid = r'data/Houston13_Trento/Target/italy6.mat'
        hsi_valid = sio.loadmat(hsi_valid)['houston18hsi']
        hsi_valid = hsi_valid.astype(np.float32) 
        lidar_valid = r'data/Houston13_Trento/Target/trento_lidar.mat'
        lidar_valid = sio.loadmat(lidar_valid)['trento_lidar'][:,:,0]
        #lidar_valid = preprocessing.scale(lidar_valid)
        lidar_valid = lidar_valid[:, :, np.newaxis]
        lidar_valid = lidar_valid.astype(np.float32) 
        label_valid = r'data/Houston13_Trento/Target/italyNew_mask_test.mat'
        label_valid = sio.loadmat(label_valid)['italynewGT']
        col_valid, row_valid = 600, 166

        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = pca_num

            del hsi_matrix
    elif dataset == 'trento_muufl':
        num_classes = 4
        band = 54
        hsi = r'data/Trento_Muufl/Source/trento_hsi_new.mat'
        hsi = sio.loadmat(hsi)['trento_hsi']
        hsi = hsi.astype(np.float32) 
        lidar = r'data/Trento_Muufl/Source/trento_lidar.mat'
        lidar = sio.loadmat(lidar)['trento_lidar'][:,:,0]
        lidar = lidar[:, :, np.newaxis]
        lidar = lidar.astype(np.float32) 
        label = r'data/Trento_Muufl/Source/trento_gt_cd.mat'
        label = sio.loadmat(label)['gt_cd']
        col_train, row_train = 600, 166

        hsi_valid = r'data/Trento_Muufl/Target/muffl_hsi_new.mat'
        hsi_valid = sio.loadmat(hsi_valid)['muufl_hsi']
        hsi_valid = hsi_valid.astype(np.float32) 
        lidar_valid = r'data/Trento_Muufl/Target/muffl_lidar.mat'
        lidar_valid = sio.loadmat(lidar_valid)['data'][:,:,0]
        lidar_valid = lidar_valid[:, :, np.newaxis]
        lidar_valid = lidar_valid.astype(np.float32) 
        label_valid = r'data/Trento_Muufl/Target/muufl_gt_cd.mat'
        label_valid = sio.loadmat(label_valid)['gt_cd']
        col_valid, row_valid = 220, 325

        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            pca = PCA(n_components=pca_num)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*10
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 10
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = pca_num

            del hsi_matrix

    else:
        raise ValueError("Unknown dataset")

    # normalize data
    if band_norm:
        norm = band_normalization
    else:
        norm = normalization
    
    if dataset == 'nashua_hanover' or dataset == 'houston1318' or dataset == 'trento_muufl':
        hsi = norm(hsi)
        lidar = norm(lidar)
        hsi_valid = norm(hsi_valid)
        lidar_valid = norm(lidar_valid)
    elif dataset == 'augsburg_berlin':
        hsi = norm(hsi)
        lidar = norm(lidar)
        hsi_valid = norm(hsi_valid)
        lidar_valid = norm(lidar_valid)
    elif dataset == 'houston13_trento':
        #hsi = norm(hsi)
        #lidar = norm(lidar)
        #hsi_valid = norm(hsi_valid)
        lidar_valid = norm(lidar_valid)
    else:
        raise ValueError("Unknown dataset")
        
    band_lidar = lidar.shape[2]
    
    return hsi, lidar, label, hsi_valid, lidar_valid, label_valid, num_classes, band, band_lidar


def slide_crop(dataset, patch, overlay, pca_flag=False, pca_num = 10, band_norm_flag=False, aug_flag=False):
    hsi, lidar, label, hsi_valid, lidar_valid, label_valid, num_classes, band, band_lidar = read_data(dataset, pca_flag, pca_num,
                                                                                                      band_norm_flag)

    if dataset == 'augsburg_berlin':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225
        
    elif dataset == 'nashua_hanover':
        col_train, row_train = 1050, 450
        col_valid, row_valid = 620, 700
        
    elif dataset == 'houston1318':
        col_train, row_train = 954, 210
        col_valid, row_valid = 954, 210  
    elif dataset == 'houston13_trento':
        col_train, row_train = 954, 210
        col_valid, row_valid = 600, 166  
    elif dataset == 'trento_muufl':
        col_train, row_train = 600, 166
        col_valid, row_valid = 220, 325  

    else:
        raise ValueError("Unknown dataset")

    transform = iaa.Sequential([
        iaa.Rot90([0, 1, 2, 3]),
        iaa.VerticalFlip(p=0.5),
        iaa.HorizontalFlip(p=0.5),
    ])

    # slide crop for train data
    if dataset=='augsburg' or dataset == 'beijing':
        window_set_train = sw.generate(hsi, sw.DimOrder.HeightWidthChannel, patch, overlay)
        hsi_list = []
        msi_list = []
        sar_list = []
        label_list = []
        for window in window_set_train:
            subset_hsi = hsi[window.indices()]
            subset_msi = msi[window.indices()]
            subset_sar = sar[window.indices()]
            subset_label = label[window.indices()]
            if aug_flag:
                all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
                img1, label1 = transform(image=all_img,
                                         segmentation_maps=np.stack(
                                             (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                             , axis=-1).astype(np.int32))

                subset_label = label1[0, :, :, 0]
                # subset_hsi = img1[:, :, :10]
                # subset_msi = img1[:, :, 10:14]
                # subset_sar = img1[:, :, 14:16]
                subset_hsi = img1[:, :, :band]
                subset_msi = img1[:, :, band:(band+4)]
                subset_sar = img1[:, :, (band+4):(band+4+2)]

            hsi_list.append(subset_hsi)
            msi_list.append(subset_msi)
            sar_list.append(subset_sar)
            label_list.append(subset_label)
        del hsi, msi, sar, label, subset_hsi, subset_msi, subset_sar, subset_label, img1, label1, window_set_train
        hsi_list = np.array(hsi_list)
        msi_list = np.array(msi_list)
        sar_list = np.array(sar_list)
        label_list = np.array(label_list)
        # has_zero = np.any(label_list==0)
        # print(has_zero)
        #print(11111111111, hsi_list.shape)#140

        # non-overlay crop for valid data
        window_set_valid = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, overlay)
        hsi_valid_list = []
        msi_valid_list = []
        sar_valid_list = []
        label_valid_list = []
        for window in window_set_valid:
            subset_hsi = hsi_valid[window.indices()]
            subset_msi = msi_valid[window.indices()]
            subset_sar = sar_valid[window.indices()]
            subset_label = label_valid[window.indices()]
            if aug_flag:
                all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
                img1, label1 = transform(image=all_img,
                                         segmentation_maps=np.stack(
                                             (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                             , axis=-1).astype(np.int32))

                subset_label = label1[0, :, :, 0]
                # subset_hsi = img1[:, :, :10]
                # subset_msi = img1[:, :, 10:14]
                # subset_sar = img1[:, :, 14:16]
                subset_hsi = img1[:, :, :band]
                subset_msi = img1[:, :, band:(band+4)]
                subset_sar = img1[:, :, (band+4):(band+4+2)]
                
            hsi_valid_list.append(subset_hsi)
            msi_valid_list.append(subset_msi)
            sar_valid_list.append(subset_sar)
            label_valid_list.append(subset_label)
        hsi_valid_list = np.array(hsi_valid_list)
        msi_valid_list = np.array(msi_valid_list)
        sar_valid_list = np.array(sar_valid_list)
        label_valid_list = np.array(label_valid_list)

        # non-overlay crop for valid data
        window_set_test = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
        hsi_test_list = []
        msi_test_list = []
        sar_test_list = []
        label_test_list = []
        for window in window_set_test:
            subset_hsi = hsi_valid[window.indices()]
            subset_msi = msi_valid[window.indices()]
            subset_sar = sar_valid[window.indices()]
            subset_label = label_valid[window.indices()]
            hsi_test_list.append(subset_hsi)
            msi_test_list.append(subset_msi)
            sar_test_list.append(subset_sar)
            label_test_list.append(subset_label)
        del hsi_valid, msi_valid, sar_valid, label_valid, subset_hsi, subset_msi, subset_sar, subset_label, window_set_valid
        hsi_test_list = np.array(hsi_test_list)
        msi_test_list = np.array(msi_test_list)
        sar_test_list = np.array(sar_test_list)
        label_test_list = np.array(label_test_list)
        #print(2222222222, hsi_test_list.shape)#

        # construct dataset
        hsi_list = torch.from_numpy(hsi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        msi_list = torch.from_numpy(msi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        sar_list = torch.from_numpy(sar_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_list = torch.from_numpy(label_list).type(torch.LongTensor)
        label_train = Data.TensorDataset(hsi_list, msi_list, sar_list, label_list)

        hsi_valid_list = torch.from_numpy(hsi_valid_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        msi_valid_list = torch.from_numpy(msi_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        sar_valid_list = torch.from_numpy(sar_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
        label_valid = Data.TensorDataset(hsi_valid_list, msi_valid_list, sar_valid_list, label_valid_list)

        hsi_test_list = torch.from_numpy(hsi_test_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        msi_test_list = torch.from_numpy(msi_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        sar_test_list = torch.from_numpy(sar_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_test_list = torch.from_numpy(label_test_list).type(torch.LongTensor)
        label_test = Data.TensorDataset(hsi_test_list, msi_test_list, sar_test_list, label_test_list)

    # slide crop for train data
    elif dataset=='nashua_hanover' or dataset=='houston1318' or dataset=='augsburg_berlin' or dataset=='houston13_trento' or dataset=='trento_muufl':
        window_set_train = sw.generate(hsi, sw.DimOrder.HeightWidthChannel, patch, overlay)
        hsi_list = []
        lidar_list = []
        label_list = []
        for window in window_set_train:
            subset_hsi = hsi[window.indices()]
            subset_lidar = lidar[window.indices()]
            subset_label = label[window.indices()]
            if aug_flag:
                all_img = np.concatenate((subset_hsi, subset_lidar), axis=-1)
                img1, label1 = transform(image=all_img,
                                         segmentation_maps=np.stack(
                                             (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                             , axis=-1).astype(np.int32))

                subset_label = label1[0, :, :, 0]
                # subset_hsi = img1[:, :, :10]
                # subset_msi = img1[:, :, 10:14]
                # subset_sar = img1[:, :, 14:16]
                subset_hsi = img1[:, :, :band]
                subset_lidar = img1[:, :, band:(band+lidar.shape[2])]

            hsi_list.append(subset_hsi)
            lidar_list.append(subset_lidar)
            label_list.append(subset_label)
        del hsi, lidar, label, subset_hsi, subset_lidar, subset_label, img1, label1, window_set_train
        hsi_list = np.array(hsi_list)
        lidar_list = np.array(lidar_list)
        label_list = np.array(label_list)
        
        # print(111111111111111, label_list.shape, hsi_list.shape)
        zero_matrix = np.zeros((label_list.shape[0], num_classes, label_list.shape[1], label_list.shape[2]))
        
        for n in range(label_list.shape[0]):
            # print(222222222222222, label_list[i])
            for i in range(num_classes):
                for j in range(label_list.shape[1]):
                        for k in range(label_list.shape[2]):
                            if label_list[n, j, k] == i+1:
                                zero_matrix[n, :, j, k]=i+1
                            # print(22222222222222222, zero_matrix[n, j, k, :])
        label_GAN = zero_matrix
        
        
        # has_zero = np.any(label_list==0)
        # print(has_zero)
        #print(11111111111, hsi_list.shape)#140

        # non-overlay crop for valid data
        window_set_valid = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, overlay)
        hsi_valid_list = []
        lidar_valid_list = []
        label_valid_list = []
        for window in window_set_valid:
            subset_hsi = hsi_valid[window.indices()]
            subset_lidar = lidar_valid[window.indices()]
            subset_label = label_valid[window.indices()]
            if aug_flag:
                all_img = np.concatenate((subset_hsi, subset_lidar), axis=-1)
                img1, label1 = transform(image=all_img,
                                         segmentation_maps=np.stack(
                                             (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                             , axis=-1).astype(np.int32))

                subset_label = label1[0, :, :, 0]
                # subset_hsi = img1[:, :, :10]
                # subset_msi = img1[:, :, 10:14]
                # subset_sar = img1[:, :, 14:16]
                subset_hsi = img1[:, :, :band]
                subset_lidar = img1[:, :, band:(band+lidar_valid.shape[2])]
                
            hsi_valid_list.append(subset_hsi)
            lidar_valid_list.append(subset_lidar)
            label_valid_list.append(subset_label)
        hsi_valid_list = np.array(hsi_valid_list)
        lidar_valid_list = np.array(lidar_valid_list)
        label_valid_list = np.array(label_valid_list)

        # non-overlay crop for valid data
        window_set_test = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
        hsi_test_list = []
        lidar_test_list = []
        label_test_list = []
        for window in window_set_test:
            subset_hsi = hsi_valid[window.indices()]
            subset_lidar = lidar_valid[window.indices()]
            subset_label = label_valid[window.indices()]
            hsi_test_list.append(subset_hsi)
            lidar_test_list.append(subset_lidar)
            label_test_list.append(subset_label)
        del hsi_valid, lidar_valid, label_valid, subset_hsi, subset_lidar, subset_label, window_set_valid
        hsi_test_list = np.array(hsi_test_list)
        lidar_test_list = np.array(lidar_test_list)
        label_test_list = np.array(label_test_list)
        #print(2222222222, hsi_test_list.shape)#

        # construct dataset
        hsi_list = torch.from_numpy(hsi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        lidar_list = torch.from_numpy(lidar_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_list = torch.from_numpy(label_list).type(torch.LongTensor)
        # label_train = Data.TensorDataset(hsi_list, lidar_list, label_list)
        
        label_GAN = torch.from_numpy(label_GAN).type(torch.LongTensor)
        label_train = Data.TensorDataset(hsi_list, lidar_list, label_list, label_GAN)

        hsi_valid_list = torch.from_numpy(hsi_valid_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        lidar_valid_list = torch.from_numpy(lidar_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
        label_valid = Data.TensorDataset(hsi_valid_list, lidar_valid_list, label_valid_list)

        hsi_test_list = torch.from_numpy(hsi_test_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        lidar_test_list = torch.from_numpy(lidar_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
        label_test_list = torch.from_numpy(label_test_list).type(torch.LongTensor)
        label_test = Data.TensorDataset(hsi_test_list, lidar_test_list, label_test_list)
    else:
        raise ValueError("Unknown dataset")


    return label_train, label_valid, label_test, num_classes, band, band_lidar


def slide_crop_all_modalities(dataset, patch, overlay):
    hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band = read_data(dataset)


    if dataset == 'augsburg':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")

    # slide crop for train data
    img = np.concatenate([hsi, msi, sar], axis=2)  # w, h, c
    del hsi, msi, sar
    window_set_train = sw.generate(img, sw.DimOrder.HeightWidthChannel, patch, overlay)
    img_list = []
    label_list = []
    for window in window_set_train:
        subset_img = img[window.indices()]
        subset_label = label[window.indices()]
        img_list.append(subset_img)
        label_list.append(subset_label)
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    del img, label

    # non-overlay crop for valid data
    img_valid = np.concatenate([hsi_valid, msi_valid, sar_valid], axis=2)
    del hsi_valid, msi_valid, sar_valid

    window_set_valid = sw.generate(img_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    img_valid_list = []
    label_valid_list = []
    for window in window_set_valid:
        subset_img = img_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        img_valid_list.append(subset_img)
        label_valid_list.append(subset_label)
    img_valid_list = np.array(img_valid_list)
    label_valid_list = np.array(label_valid_list)
    del img_valid, label_valid

    # construct dataset
    img_list = torch.from_numpy(img_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_list = torch.from_numpy(label_list).type(torch.LongTensor)
    label_train = Data.TensorDataset(img_list, label_list)

    img_valid_list = torch.from_numpy(img_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
    label_valid = Data.TensorDataset(img_valid_list, label_valid_list)

    return label_train, label_valid, num_classes, band
