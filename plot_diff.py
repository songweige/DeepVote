import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy import misc
import imageio as misc
from shutil import copyfile

def getImageMinMax(im_np):
    return np.nanpercentile(im_np, 10), np.nanpercentile(im_np, 90)


def getColorMapFromPalette(im_np, palette, im_min = None, im_max = None):
    if (im_min is None):
        im_min, im_max = getImageMinMax(im_np)
    print('min:', im_min, 'max:', im_max)
    normalized = ((im_np - im_min) / (im_max - im_min))
    normalized[normalized < 0] = 0
    normalized[normalized > 1] = 1
    normalized_nan = np.isnan(normalized)
    normalized_not_nan = np.logical_not(normalized_nan)
    color_indexes = np.round(normalized * (palette.shape[0] - 2) + 1).astype('uint32')
    color_indexes[normalized_nan] = 0
    color_map = palette[color_indexes]
    return color_map

def fill(inh, gth, completeness=False):
    input_h = np.copy(inh)
    gt_h = np.copy(gth)
    input_mask = np.isnan(input_h)
    truth_mask = gt_h < 0
    count = truth_mask.shape[0]*truth_mask.shape[1]-np.sum(truth_mask)
    input_h[input_mask] = 0.
    gt_h[truth_mask] = 0.
    if completeness:
        input_h[input_mask] = -10.
    return input_h, gt_h, count


def RSME(inh, gth):
    input_h, gt_h, count = fill(inh, gth)
    # print('undefined: ', np.sum(mask), 'count: ', count)
    # return np.sum((input_h-gt_h)**2)
    return np.sqrt(np.sum((input_h-gt_h)**2)/count)


def L1E(inh, gth):
    input_h, gt_h, count = fill(inh, gth)
    # print('undefined: ', np.sum(mask), 'count: ', count)
    # return np.sum((input_h-gt_h)**2)
    return np.sum(np.abs(input_h-gt_h))/count

def accuracy(inh, gth):
    input_h, gt_h, count = fill(inh, gth)
    return np.median(np.abs(input_h-gt_h)).flatten()


def completeness(inh, gth):
    truth_mask = gth > 0
    input_h, gt_h, count = fill(inh, gth, completeness=True)
    cpm_count = np.sum(np.abs(input_h[truth_mask]-gt_h[truth_mask])<1)
    return cpm_count/count

n_pair = 6
x_len = 1001
y_len = 1001
save_fig = False
cal_baseline = False
mode = 'test'
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
result_path = '../results/eval_res_fold0_%s'%mode
data_path = '../data/'
input_path = '/home/songweig/LockheedMartin/data/MVS'
gt_path = '/home/songweig/LockheedMartin/data/DSM'

dv_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain/reconstruction_%s'%mode
dv_mean_path = '/home/songweig/LockheedMartin/CDSAlign/results/mean/reconstruction_%s'%mode
# dv_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain_res/reconstruction_%s'%mode
# dv_mean_path = '/home/songweig/LockheedMartin/CDSAlign/results/mean_res/reconstruction_%s'%mode
# dv_max_path = '/home/songweig/LockheedMartin/CDSAlign/results/large_max/reconstruction'


fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
error_palette = misc.imread(os.path.join(data_path, 'error_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]
filename = 'dem_5_35'


pair_data = []
for i in range(n_pair):
    height = np.load(os.path.join(input_path, filename, input_temp%i))
    pair_data.append(height)


mean_height = np.mean(pair_data, 0)
color_map = getColorMapFromPalette(mean_height, fire_palette)
misc.imsave(out_path+'_mean_height.png', color_map)

dv_data = np.load(os.path.join(dv_path, filename+'.npy'))
color_map = getColorMapFromPalette(dv_data-mean_height, confidence_palette)
misc.imsave(out_path+'_res.png', color_map)
