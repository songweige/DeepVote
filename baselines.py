import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as misc
from shutil import copyfile

from eval_util import *

n_pair = 6
x_len = 1001
y_len = 1001
save_fig = False
cal_baseline = False
mode = 'train'
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
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]

consense_scores = [[], [], [], []]
median_scores = [[], [], [], []]
kmedian_scores = [[], [], [], []]
dv_scores = [[], [], [], []]
dv_mean_scores = [[], [], [], []]
dv_max_scores = [[], [], [], []]

count = 0
filenames = [fn[:-4] for fn in os.listdir(dv_path)]
for filename in filenames:
    plt.clf()
    out_path = os.path.join(result_path, filename)
    # if not os.path.exists(out_path):
    #     os.mkdir(out_path)
    # if os.path.exists(out_path+'_input_color0.png'):
    #     continue
    # count += 1
    # if not os.path.exists(os.path.join(input_path, filename)):
    #     continue
    # print(count, filename+':\n')

    pair_data = []
    for i in range(n_pair):
        height = np.load(os.path.join(input_path, filename, input_temp%i))
        if save_fig:
            color_map = getColorMapFromPalette(height, fire_palette)
            misc.imsave(out_path+'_input_height%d.png'%i, color_map)
            copyfile(os.path.join(input_path, filename, input_img_temp%i), out_path+'_input_color%d.png'%i)
        pair_data.append(height)
    pair_data = np.array(pair_data)
    # load gt
    gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
    if save_fig:
        color_map = getColorMapFromPalette(gt_data, fire_palette)
        misc.imsave(out_path+'_gt_data.png', color_map)

    # consense vote

    # con_data = np.zeros((y_len, x_len))
    # for i in range(y_len):
    #     for j in range(x_len):
    con_data = np.mean(pair_data, 0)
    # con_data = np.load(os.path.join(input_path, filename, 'test.npz'))['f_infos'][:, :, 1]
    if save_fig:
        color_map = getColorMapFromPalette(con_data, fire_palette)
        misc.imsave(out_path+'_consense.png', color_map)
    con_data_rsme = RSME(con_data, gt_data)
    con_data_acc = accuracy(con_data, gt_data)
    con_data_com = completeness(con_data, gt_data)
    con_data_l1e = L1E(con_data, gt_data)
    consense_scores[0].append(con_data_rsme)
    consense_scores[1].append(con_data_acc)
    consense_scores[2].append(con_data_com)
    consense_scores[3].append(con_data_l1e)
    print('RMSE of consense vote = %.4f'%con_data_rsme)
    print('Accuracy of consense vote = %.4f'%con_data_acc)
    print('Completeness of consense vote = %.4f'%con_data_com)
    print('L1 Error of consense vote = %.4f\n'%con_data_l1e)


    if cal_baseline:
        # median fusion
        median_fusion = np.zeros((y_len, x_len))
        for i in range(y_len):
            for j in range(x_len):
                median_fusion[i, j] = np.median(pair_data[:, i, j])
        if save_fig:
            color_map = getColorMapFromPalette(median_fusion, fire_palette)
            misc.imsave(out_path+'_median_fusion.png', color_map)
        median_fusion_rsme = RSME(median_fusion, gt_data)
        median_fusion_acc = accuracy(median_fusion, gt_data)
        median_fusion_com = completeness(median_fusion, gt_data)
        median_fusion_l1e = L1E(median_fusion, gt_data)
        median_scores[0].append(median_fusion_rsme)
        median_scores[1].append(median_fusion_acc)
        median_scores[2].append(median_fusion_com)
        median_scores[3].append(median_fusion_l1e)
        print('RMSE of median fusion = %.4f'%median_fusion_rsme)
        print('Accuracy of median fusion = %.4f'%median_fusion_acc)
        print('Completeness of median fusion = %.4f'%median_fusion_com)
        print('L1 Error of median fusion = %.4f\n'%median_fusion_l1e)


        # k-medians
        kmedian_fusion = np.zeros((y_len, x_len))
        for i in range(y_len):
            for j in range(x_len):
                heights = pair_data[:, i, j]
                mask = np.logical_not(np.logical_or(np.isnan(heights), heights<0))
                # import ipdb;ipdb.set_trace()
                if sum(mask) == 0:
                    kmedian_fusion[i, j] = np.nan
                    continue
                hmax, hmin = np.max(heights[mask]), np.min(heights[mask])
                max_set = []
                min_set = []
                for item in heights[mask]:
                    max_dis, min_dis = hmax-item, item-hmin
                    if max_dis>min_dis:
                        min_set.append(item)
                    else:
                        max_set.append(item)
                if len(max_set) < 2 or len(min_set) < 2:
                    kmedian_fusion[i, j] = np.median(heights[mask])
                else:
                    dis = np.median(max_set)-np.median(min_set)
                    if dis > 5.:
                        kmedian_fusion[i, j] = np.median(max_set)
                    else:
                        kmedian_fusion[i, j] = np.median(heights[mask])
                # import ipdb;ipdb.set_trace()
        if save_fig:
            color_map = getColorMapFromPalette(kmedian_fusion, fire_palette)
            misc.imsave(out_path+'_kmedian_fusion.png', color_map)
        kmedian_fusion_rsme = RSME(kmedian_fusion, gt_data)
        kmedian_fusion_acc = accuracy(kmedian_fusion, gt_data)
        kmedian_fusion_com = completeness(kmedian_fusion, gt_data)
        kmedian_fusion_l1e = L1E(kmedian_fusion, gt_data)
        kmedian_scores[0].append(kmedian_fusion_rsme)
        kmedian_scores[1].append(kmedian_fusion_acc)
        kmedian_scores[2].append(kmedian_fusion_com)
        kmedian_scores[3].append(kmedian_fusion_l1e)
        print('RMSE of kmedian_fusion vote = %.4f'%kmedian_fusion_rsme)
        print('Accuracy of kmedian_fusion vote = %.4f'%kmedian_fusion_acc)
        print('Completeness of kmedian_fusion vote = %.4f'%kmedian_fusion_com)
        print('L1 Error of kmedian_fusion vote = %.4f\n'%kmedian_fusion_l1e)

    # deep vote
    dv_data = np.load(os.path.join(dv_path, filename+'.npy'))
    if save_fig:
        color_map = getColorMapFromPalette(dv_data, fire_palette)
        misc.imsave(out_path+'_dv.png', color_map)
    dv_data_rsme = RSME(dv_data, gt_data)
    dv_data_acc = accuracy(dv_data, gt_data)
    dv_data_com = completeness(dv_data, gt_data)
    dv_data_l1e = L1E(dv_data, gt_data)
    dv_scores[0].append(dv_data_rsme)
    dv_scores[1].append(dv_data_acc)
    dv_scores[2].append(dv_data_com)
    dv_scores[3].append(dv_data_l1e)
    print('RMSE of deep vote = %.4f'%dv_data_rsme)
    print('Accuracy of deep vote = %.4f'%dv_data_acc)
    print('Completeness of deep vote = %.4f'%dv_data_com)
    print('L1 Error of deep vote = %.4f\n'%dv_data_l1e)

    if dv_data_rsme > con_data_rsme:
        print('not good!\n')

    # deep vote with mean pooling
    dv_mean_data = np.load(os.path.join(dv_mean_path, filename+'.npy'))
    if save_fig:
        color_map = getColorMapFromPalette(dv_mean_data, fire_palette)
        misc.imsave(out_path+'_dv_mean.png', color_map)
    dv_mean_data_rsme = RSME(dv_mean_data, gt_data)
    dv_mean_data_acc = accuracy(dv_mean_data, gt_data)
    dv_mean_data_com = completeness(dv_mean_data, gt_data)
    dv_mean_data_l1e = L1E(dv_mean_data, gt_data)
    dv_mean_scores[0].append(dv_mean_data_rsme)
    dv_mean_scores[1].append(dv_mean_data_acc)
    dv_mean_scores[2].append(dv_mean_data_com)
    dv_mean_scores[3].append(dv_mean_data_l1e)
    # import ipdb;ipdb.set_trace()    
    print('RMSE of deep vote with mean pooling = %.4f'%dv_mean_data_rsme)
    print('Accuracy of deep vote with mean pooling = %.4f'%dv_mean_data_acc)
    print('Completeness of deep vote with mean pooling = %.4f'%dv_mean_data_com)
    print('L1 Error of deep vote with mean pooling = %.4f\n'%dv_mean_data_l1e)

    # deep vote with max pooling
    # dv_max_data = np.load(os.path.join(dv_max_path, filename+'.npy'))
    # if save_fig:
    # color_map = getColorMapFromPalette(dv_max_data, fire_palette)
    # misc.imsave(out_path+'_dv_max.png', color_map)
    # dv_max_data_rsme = RSME(dv_max_data, gt_data)
    # dv_max_data_acc = accuracy(dv_max_data, gt_data)
    # dv_max_data_com = completeness(dv_max_data, gt_data)
    # dv_max_scores[0].append(dv_max_data_rsme)
    # dv_max_scores[1].append(dv_max_data_acc)
    # dv_max_scores[2].append(dv_max_data_com)
    # print('RMSE of deep vote with max pooling = %.4f'%dv_max_data_rsme)
    # print('Accuracy of deep vote with max pooling = %.4f'%dv_max_data_acc)
    # print('Completeness of deep vote with max pooling = %.4f\n'%dv_max_data_com)

print('Final RMSE of consense vote = %.4f, std = %.4f'%(np.mean(consense_scores[0]), np.std(consense_scores[0])))
print('Final Accuracy of consense vote = %.4f, std = %.4f'%(np.mean(consense_scores[1]), np.std(consense_scores[1])))
print('Final Completeness of consense vote = %.4f, std = %.4f'%(np.mean(consense_scores[2]), np.std(consense_scores[2])))
print('Final L1 Error of consense vote = %.4f, std = %.4f\n'%(np.mean(consense_scores[3]), np.std(consense_scores[3])))

if cal_baseline:
    print('Final RMSE of median fusion = %.4f, std = %.4f'%(np.mean(median_scores[0]), np.std(median_scores[0])))
    print('Final Accuracy of median fusion = %.4f, std = %.4f'%(np.mean(median_scores[1]), np.std(median_scores[1])))
    print('Final Completeness of median fusion = %.4f, std = %.4f'%(np.mean(median_scores[2]), np.std(median_scores[2])))
    print('Final L1 Error of median fusion = %.4f, std = %.4f\n'%(np.mean(median_scores[3]), np.std(median_scores[3])))

    print('Final RMSE of kmedian_fusion vote = %.4f, std = %.4f'%(np.mean(kmedian_scores[0]), np.std(kmedian_scores[0])))
    print('Final Accuracy of kmedian_fusion vote = %.4f, std = %.4f'%(np.mean(kmedian_scores[1]), np.std(kmedian_scores[1])))
    print('Final Completeness of kmedian_fusion vote = %.4f, std = %.4f'%(np.mean(kmedian_scores[2]), np.std(kmedian_scores[2])))
    print('Final L1 Error of kmedian_fusion vote = %.4f, std = %.4f\n'%(np.mean(kmedian_scores[3]), np.std(kmedian_scores[3])))

print('Final RMSE of deep vote = %.4f, std = %.4f'%(np.mean(dv_scores[0]), np.std(dv_scores[0])))
print('Final Accuracy of deep vote = %.4f, std = %.4f'%(np.mean(dv_scores[1]), np.std(dv_scores[1])))
print('Final Completeness of deep vote = %.4f, std = %.4f'%(np.mean(dv_scores[2]), np.std(dv_scores[2])))
print('Final L1 Error of deep vote = %.4f, std = %.4f\n'%(np.mean(dv_scores[3]), np.std(dv_scores[3])))

print('Final RMSE of deep vote with mean pooling = %.4f, std = %.4f'%(np.mean(dv_mean_scores[0]), np.std(dv_mean_scores[0])))
print('Final Accuracy of deep vote with mean pooling = %.4f, std = %.4f'%(np.mean(dv_mean_scores[1]), np.std(dv_mean_scores[1])))
print('Final Completeness of deep vote with mean pooling = %.4f, std = %.4f'%(np.mean(dv_mean_scores[2]), np.std(dv_mean_scores[2])))
print('Final L1 Error of deep vote with mean pooling = %.4f, std = %.4f\n'%(np.mean(dv_mean_scores[3]), np.std(dv_mean_scores[3])))

# print('Final RMSE of deep vote with max pooling = %.4f'%np.mean(dv_max_scores[0]))
# print('Final Accuracy of deep vote with max pooling = %.4f'%np.mean(dv_max_scores[1]))
# print('Final Completeness of deep vote with max pooling = %.4f\n'%np.mean(dv_max_scores[2]))