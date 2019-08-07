import os
import time
import numpy as np
from scipy import misc



data_path = '/home/songweig/LockheedMartin/data/MVS'
gt_path = '/home/songweig/LockheedMartin/data/DSM'
n_pair = 6
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
filenames = os.listdir(data_path)
# N x P x W x H x 2
begin_time = time.time()
for filename in filenames:
    if filename != 'dem_10_20':
        continue
    file_tmp = []
    for i in range(n_pair):
        height = np.load(os.path.join(data_path, filename, input_temp%i))
        mask = np.isnan(height)
        height[mask] = 0
        img = misc.imread(os.path.join(data_path, filename, input_img_temp%i))
        file_tmp.append(np.dstack((img, height)).transpose(2, 0, 1))
    np.save(os.path.join(data_path, filename, 'trainX'), np.array(file_tmp))

print(time.time()-begin_time)