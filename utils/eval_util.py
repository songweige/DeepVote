"""
Utils related to evaluation like metric calculation, visualization
"""
import os
import numpy as np

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
    return np.sqrt(np.sum((input_h-gt_h)**2)/count)


def L1E(inh, gth):
    input_h, gt_h, count = fill(inh, gth)
    return np.sum(np.abs(input_h-gt_h))/count

def accuracy(inh, gth):
    input_h, gt_h, count = fill(inh, gth)
    return np.median(np.abs(input_h-gt_h)).flatten()

def completeness(inh, gth):
    truth_mask = gth > 0
    input_h, gt_h, count = fill(inh, gth, completeness=True)
    cpm_count = np.sum(np.abs(input_h[truth_mask]-gt_h[truth_mask])<1).astype(np.float)
    return cpm_count/count

def evaluate(input_data, gt_data):
    rsme_metric = RSME(input_data, gt_data)
    acc_metric = accuracy(input_data, gt_data)
    com_metric = completeness(input_data, gt_data)
    l1e_metric = L1E(input_data, gt_data)
    return rsme_metric, acc_metric, com_metric, l1e_metric