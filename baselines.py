import os
import numpy as np
import imageio as misc
from shutil import copyfile

import eval_util

def partial_median(x):
    k = x.shape[0]//2
    med = np.median(x[:k][x[k:].astype(bool)])
    return med

n_pair = 6
save_fig = False
mode = 'test'
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
result_path = '../results/eval_all_%s'%mode
data_path = '../data/'
input_path = '/home/songweig/LockheedMartin/data/MVS'
gt_path = '/home/songweig/LockheedMartin/data/DSM'

dv_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain_res/reconstruction_%s'%mode
dv_mean_path = '/home/songweig/LockheedMartin/CDSAlign/results/mean_res/reconstruction_%s'%mode
dv_mean2_path = '/home/songweig/LockheedMartin/CDSAlign/results/mean2/reconstruction_%s'%mode
dv_large_path = '/home/songweig/LockheedMartin/CDSAlign/results/large/reconstruction_%s'%mode
dv_weighted_path = '/home/songweig/LockheedMartin/CDSAlign/results/weighted/reconstruction_%s'%mode

if not os.path.exists(result_path):
    os.mkdir(result_path)

fire_palette = misc.imread(os.path.join('image/fire_palette.png'))[0][:, 0:3]

consense_scores = []
median_fusion_scores = []
mean_fusion_scores = []
kmedian_fusion_scores = []
dv_scores = []
dv_mean_scores = []
dv_mean2_scores = []
dv_large_scores = []
dv_weighted_scores = []

filenames = [fn[:-4] for fn in os.listdir(dv_path)]
print('Number of test data: %d'%len(filenames))
for filename in filenames:
    out_path = os.path.join(result_path, filename)
    print(filename+':\n')
    pair_data = []
    for i in range(n_pair):
        height = np.load(os.path.join(input_path, filename, input_temp%i))
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(height, fire_palette)
            misc.imsave(out_path+'_input_height%d.png'%i, color_map)
            copyfile(os.path.join(input_path, filename, input_img_temp%i), out_path+'_input_color%d.png'%i)
        pair_data.append(height)
    pair_data = np.array(pair_data)

    # load gt
    gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
        misc.imsave(out_path+'_gt_data.png', color_map)

    # consense vote
    con_data = np.load(os.path.join(input_path, filename, 'test.npz'))['f_infos'][:, :, 1]
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(con_data, fire_palette)
        misc.imsave(out_path+'_consense.png', color_map)
    metrics = eval_util.evaluate(con_data, gt_data)
    consense_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of consense vote: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # mean fusion
    mean_data = np.mean(pair_data, 0)
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(mean_data, fire_palette)
        misc.imsave(out_path+'_mean.png', color_map)
    metrics = eval_util.evaluate(mean_data, gt_data)
    mean_fusion_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of mean fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # median fusion
    median_fusion = np.median(pair_data, 0)
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(median_fusion, fire_palette)
        misc.imsave(out_path+'_median_fusion.png', color_map)
    metrics = eval_util.evaluate(median_fusion, gt_data)
    median_fusion_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of median fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

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
    max_modal_data = np.vstack((pair_data, max_set)).reshape(n_pair*2, -1)
    kmedian_fusion[bi_modal_ids] = np.apply_along_axis(partial_median, 0, max_modal_data[:, bi_modal_ids.reshape(-1)])
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(kmedian_fusion, fire_palette)
        misc.imsave(out_path+'_kmedian_fusion.png', color_map)
    metrics = eval_util.evaluate(kmedian_fusion, gt_data)
    kmedian_fusion_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of kmedian fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # deep vote
    dv_data = np.load(os.path.join(dv_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_data, fire_palette)
        misc.imsave(out_path+'_dv.png', color_map)
    metrics = eval_util.evaluate(dv_data, gt_data)
    dv_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of dv: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # deep vote with mean pooling
    dv_mean_data = np.load(os.path.join(dv_mean_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_mean_data, fire_palette)
        misc.imsave(out_path+'_dv_mean.png', color_map)
    metrics = eval_util.evaluate(dv_mean_data, gt_data)
    dv_mean_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of dv_mean: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # deep vote with mean2 pooling
    dv_mean2_data = np.load(os.path.join(dv_mean2_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_mean2_data, fire_palette)
        misc.imsave(out_path+'_dv_mean2.png', color_map)
    metrics = eval_util.evaluate(dv_mean2_data, gt_data)
    dv_mean2_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of dv_mean2: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # large deep vote
    dv_large_data = np.load(os.path.join(dv_large_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_large_data, fire_palette)
        misc.imsave(out_path+'_dv_large.png', color_map)
    metrics = eval_util.evaluate(dv_mean_data, gt_data)
    dv_large_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of dv_large fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

    # weighted deep vote
    dv_weighted_data = np.load(os.path.join(dv_weighted_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(dv_weighted_data, fire_palette)
        misc.imsave(out_path+'_dv_weighted.png', color_map)
    metrics = eval_util.evaluate(dv_mean_data, gt_data)
    dv_weighted_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of dv_weighted fusion: %.4f,  %.4f,  %.4f,  %.4f\n'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

print('Final RMSE, Accuracy, Completeness, L1 Error of consense vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(consense_scores, 1).tolist()+np.std(consense_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of mean fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(mean_fusion_scores, 1).tolist()+np.std(mean_fusion_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of median fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(median_fusion_scores, 1).tolist()+np.std(median_fusion_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of kmedian fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(kmedian_fusion_scores, 1).tolist()+np.std(kmedian_fusion_scores, 1).tolist())

print('Final RMSE, Accuracy, Completeness, L1 Error of deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(dv_scores, 1).tolist()+np.std(dv_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of mean deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(dv_mean_scores, 1).tolist()+np.std(dv_mean_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of mean2 deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(dv_mean2_scores, 1).tolist()+np.std(dv_mean2_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of large deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(dv_large_scores, 1).tolist()+np.std(dv_large_scores, 1).tolist())
print('Final RMSE, Accuracy, Completeness, L1 Error of weighted deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    np.mean(dv_weighted_scores, 1).tolist()+np.std(dv_weighted_scores, 1).tolist())