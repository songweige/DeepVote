"""
Data  Utils.
"""

import os
import time
import numpy as np
import imageio as misc

def save_height(outpath, hms, fns, mode):
    # hms: N x length x length
    for hm, fn in zip(hms, fns):
        np.save(os.path.join(os.path.join('../results', outpath, 'reconstruction_%s'%mode), fn+'.npy'), hm)

# load input sets and ground truth
def load_data(gt_path, data_path, n_pair=6, input_height_temp='FF-%d.npy', input_img_temp='%d-color.png'):
    # load input data
    AllX = []
    Ally = []
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    filenames = os.listdir(data_path)
    # N x P x W x H x 2
    begin_time = time.time()
    print('start to load data')
    for filename in filenames:
        input_imgs, height_gt = load_folder(data_path=os.path.join(data_path, filename), gt_path=os.path.join(gt_path, filename+'.npy'), n_pair=6, 
                    mode='train', input_height_temp='FF-%d.npy', input_img_temp='%d-color.png')
        AllX.append(input_imgs)
        Ally.append(height_gt)
    AllX, Ally, filenames = np.array(AllX), np.array(Ally), np.array(filenames)
    print('it took %.2fs to load data'%(time.time()-begin_time))
    return AllX, Ally, filenames


# load data from given file folder
def load_folder(data_path, gt_path=None, n_pair=6, mode='train', input_height_temp='FF-%d.npy', input_img_temp='%d-color.png'):
    # load input data
    AllX = []
    Ally = []
    # N x P x W x H x 2
    # load uncalibrated height map and color images
    file_tmp = []
    for i in range(n_pair):
        height = np.load(os.path.join(data_path, input_height_temp%i))
        mask = np.isnan(height)
        height[mask] = 0
        img = misc.imread(os.path.join(data_path, input_img_temp%i))
        file_tmp.append(np.dstack((img, height)).transpose(2, 0, 1))
        if mode == 'train':
            # load ground truth
            height_gt = np.load(os.path.join(gt_path))

    if mode == 'train':
        return np.array(file_tmp), height_gt
    else:
        return np.array(file_tmp)