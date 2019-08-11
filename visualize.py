import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


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


def medieanerror(hm_input, hm_gt):
    hm_gt = cv2.resize(hm_gt, (hm_input.shape[1], hm_input.shape[0]))
    print(hm_input.shape, hm_gt.shape)
    width, height = hm_input.shape
    not_nan_mask = np.logical_and(np.logical_not(np.isnan(hm_input)), np.logical_not(hm_gt<0))
    error = np.median(np.abs(hm_gt-hm_input)[not_nan_mask])
    return error, hm_gt


result_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain/test_epochs/'
data_path = '../data'
gt_path = '/home/songweig/LockheedMartin/data/DSM'

fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]

filenames = os.listdir(result_path)
for filename in filenames:
    if not filename.endswith('npy'):
        continue
    data = np.load(os.path.join(result_path, filename))
    color_map = getColorMapFromPalette(data, fire_palette, im_min = 15, im_max = 35)
    misc.imsave(result_path+'%s.png'%filename[:-4], color_map)
    if '399' in filename:
        gt_data = np.load(os.path.join(gt_path, filename[:-8]+'.npy'))
        color_map = getColorMapFromPalette(gt_data, fire_palette, im_min = 15, im_max = 35)
        misc.imsave(result_path+'%s_gt_data.png'%filename[:-4], color_map)


