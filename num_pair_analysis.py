import os
import numpy as np
import imageio as misc
from shutil import copyfile
import matplotlib.pyplot as plt

import eval_util

def partial_median(x):
    k = x.shape[0]//2
    med = np.median(x[:k][x[k:].astype(bool)])
    return med

n_pair = 6
save_fig = True
mode = 'test'
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
result_path = '../results/eval_num_pair_618_%s'%mode
data_path = '../data/'
input_path = '/home/songweig/LockheedMartin/data/MVS'
gt_path = '/home/songweig/LockheedMartin/data/DSM'

dv_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain_res/reconstruction'
filename = 'dem_6_18'


if not os.path.exists(result_path):
    os.mkdir(result_path)

gt_data = np.load(os.path.join(gt_path, filename+'.npy'))

fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]

height1 = np.load(os.path.join(input_path, filename, input_temp%0))
metrics = eval_util.evaluate(height1, gt_data)

consense_scores = [metrics[0]]
median_fusion_scores = [metrics[0]]
mean_fusion_scores = [metrics[0]]
kmedian_fusion_scores = [metrics[0]]

height1 = np.load(os.path.join(dv_path, filename+'_fold0_399_pair%d.npy'%1))
metrics = eval_util.evaluate(height1, gt_data)
dv_scores = [metrics[0]]

for n_pair in range(2, 11):
    out_path = os.path.join(result_path, filename)
    print(filename+':\n')
    pair_data = []
    for i in range(n_pair):
        height = np.load(os.path.join(input_path, filename, input_temp%i))
        pair_data.append(height)
    pair_data = np.array(pair_data)

    # load gt
    gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
        misc.imsave(out_path+'_gt_data%d.png'%n_pair, color_map)

    # consense vote
    con_data = np.load(os.path.join(input_path, filename, 'test.npz'))['f_infos'][:, :, 1]
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(con_data, fire_palette)
        misc.imsave(out_path+'_consense%d.png'%n_pair, color_map)
    metrics = eval_util.evaluate(con_data, gt_data)
    consense_scores.append(metrics[0])

    # mean fusion
    mean_data = np.mean(pair_data, 0)
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(mean_data, fire_palette)
        misc.imsave(out_path+'_consense%d.png'%n_pair, color_map)
    metrics = eval_util.evaluate(mean_data, gt_data)
    mean_fusion_scores.append(metrics[0])

    median_fusion = np.median(pair_data, 0)
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(median_fusion, fire_palette)
        misc.imsave(out_path+'_median_fusion%d.png'%n_pair, color_map)
    metrics = eval_util.evaluate(median_fusion, gt_data)
    median_fusion_scores.append(metrics[0])

    # k-medians pair_data: n x w x h
    kmedian_fusion = median_fusion
    max_map = np.max(pair_data, 0, keepdims=True)
    min_map = np.min(pair_data, 0, keepdims=True)
    dis_max_map = np.abs(pair_data-max_map)
    dis_min_map = np.abs(pair_data-min_map)
    max_set = dis_max_map>=dis_min_map
    min_set = dis_max_map<dis_min_map
    max_set_count = np.sum(max_set, 0)
    min_set_count = np.sum(min_set, 0)

    bi_modal_ids = np.logical_and((max_map[0]-min_map[0])>5, max_set_count>1, min_set_count>1)
    if np.sum(bi_modal_ids)>0:
        max_modal_data = np.vstack((pair_data, max_set)).reshape(n_pair*2, -1)
        kmedian_fusion[bi_modal_ids] = np.apply_along_axis(partial_median, 0, max_modal_data[:, bi_modal_ids.reshape(-1)])
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(kmedian_fusion, fire_palette)
        misc.imsave(out_path+'_kmedian_fusion%d.png'%n_pair, color_map)
    metrics = eval_util.evaluate(kmedian_fusion, gt_data)
    kmedian_fusion_scores.append(metrics[0])

    # deep vote
    dv_data = np.load(os.path.join(dv_path, filename+'_fold0_399_pair%d.npy'%n_pair))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_data, fire_palette)
        misc.imsave(out_path+'_dv_%d.png'%n_pair, color_map)
    metrics = eval_util.evaluate(dv_data, gt_data)
    dv_scores.append(metrics[0])


import ipdb;ipdb.set_trace()

fig, ax1 = plt.subplots(1, 1, figsize=(20,10))
n_pairs = [i+1 for i in range(10)]
ax1.plot(n_pairs, mean_fusion_scores, 'turquoise', linewidth=8.0, label='mean')
ax1.plot(n_pairs, median_fusion_scores, 'limegreen', linewidth=8.0, label='median')
ax1.plot(n_pairs, kmedian_fusion_scores, 'orange', linewidth=8.0, label='kmedian')
ax1.plot(n_pairs, dv_scores, 'firebrick', linewidth=8.0, label='deep vote')
ax1.set_xlabel('Num of pairs')
ax1.set_ylabel('RMSE')
ax1.grid(True)
ax1.legend()
plt.savefig('eval/num_pair.jpg')