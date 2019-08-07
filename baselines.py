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
save_fig = True
cal_baseline = False
mode = 'test'
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
result_path = '../results/eval_res_fold0_%s'%mode
data_path = '../data/'
input_path = '/home/songweig/LockheedMartin/data/MVS'
gt_path = '/home/songweig/LockheedMartin/data/DSM'

dv_path = '/home/songweig/LockheedMartin/CDSAlign/results/plain_res/reconstruction_%s'%mode
dv_mean_path = '/home/songweig/LockheedMartin/CDSAlign/results/mean_res/reconstruction_%s'%mode
# dv_max_path = '/home/songweig/LockheedMartin/CDSAlign/results/large_max/reconstruction'


fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]

consense_scores = [[], [], []]
median_scores = [[], [], []]
kmedian_scores = [[], [], []]
dv_scores = [[], [], []]
dv_mean_scores = [[], [], []]
dv_max_scores = [[], [], []]

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
        pair_data.append(height)
    pair_data = np.array(pair_data)
    # load gt
    gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
    if save_fig:
        copyfile(os.path.join(input_path, filename, input_img_temp%i), out_path+'_input_color0.png')
        plt.imshow(getColorMapFromPalette(gt_data, fire_palette))
        plt.axis('off')
        plt.savefig(out_path+'_gt_data.png')

    # consense vote
    con_data = np.load(os.path.join(input_path, filename, 'test.npz'))['f_infos'][:, :, 1]
    if save_fig:
        plt.imshow(getColorMapFromPalette(con_data, fire_palette))
        plt.axis('off')
        plt.savefig(out_path+'_consense.png')
    con_data_rsme = RSME(con_data, gt_data)
    con_data_acc = accuracy(con_data, gt_data)
    con_data_com = completeness(con_data, gt_data)
    consense_scores[0].append(con_data_rsme)
    consense_scores[1].append(con_data_acc)
    consense_scores[2].append(con_data_com)
    print('RMSE of consense vote = %.4f'%con_data_rsme)
    print('Accuracy of consense vote = %.4f'%con_data_acc)
    print('Completeness of consense vote = %.4f\n'%con_data_com)


    if cal_baseline:
        # median fusion
        median_fusion = np.zeros((y_len, x_len))
        for i in range(y_len):
            for j in range(x_len):
                median_fusion[i, j] = np.median(pair_data[:, i, j])
        if save_fig:
            plt.imshow(getColorMapFromPalette(median_fusion, fire_palette))
            plt.axis('off')
            plt.savefig(out_path+'_median_fusion.png')
        median_fusion_rsme = RSME(median_fusion, gt_data)
        median_fusion_acc = accuracy(median_fusion, gt_data)
        median_fusion_com = completeness(median_fusion, gt_data)
        median_scores[0].append(median_fusion_rsme)
        median_scores[1].append(median_fusion_acc)
        median_scores[2].append(median_fusion_com)
        print('RMSE of median fusion = %.4f'%median_fusion_rsme)
        print('Accuracy of median fusion = %.4f'%median_fusion_acc)
        print('Completeness of median fusion = %.4f\n'%median_fusion_com)


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
            plt.imshow(getColorMapFromPalette(kmedian_fusion, fire_palette))
            plt.axis('off')
            plt.savefig(out_path+'_kmedian_fusion.png')
        kmedian_fusion_rsme = RSME(kmedian_fusion, gt_data)
        kmedian_fusion_acc = accuracy(kmedian_fusion, gt_data)
        kmedian_fusion_com = completeness(kmedian_fusion, gt_data)
        kmedian_scores[0].append(kmedian_fusion_rsme)
        kmedian_scores[1].append(kmedian_fusion_acc)
        kmedian_scores[2].append(kmedian_fusion_com)
        print('RMSE of kmedian_fusion vote = %.4f'%kmedian_fusion_rsme)
        print('Accuracy of kmedian_fusion vote = %.4f'%kmedian_fusion_acc)
        print('Completeness of kmedian_fusion vote = %.4f\n'%kmedian_fusion_com)

    # deep vote
    dv_data = np.load(os.path.join(dv_path, filename+'.npy'))
    if save_fig:
        plt.imshow(getColorMapFromPalette(dv_data, fire_palette))
        plt.axis('off')
        plt.savefig(out_path+'_dv.png')
    dv_data_rsme = RSME(dv_data, gt_data)
    dv_data_acc = accuracy(dv_data, gt_data)
    dv_data_com = completeness(dv_data, gt_data)
    dv_scores[0].append(dv_data_rsme)
    dv_scores[1].append(dv_data_acc)
    dv_scores[2].append(dv_data_com)
    print('RMSE of deep vote = %.4f'%dv_data_rsme)
    print('Accuracy of deep vote = %.4f'%dv_data_acc)
    print('Completeness of deep vote = %.4f\n'%dv_data_com)

    if dv_data_rsme > con_data_rsme:
        print('not good!\n')

    # deep vote with mean pooling
    dv_mean_data = np.load(os.path.join(dv_mean_path, filename+'.npy'))
    if save_fig:
        plt.imshow(getColorMapFromPalette(dv_mean_data, fire_palette))
        plt.axis('off')
        plt.savefig(out_path+'_dv_mean.png')
    dv_mean_data_rsme = RSME(dv_mean_data, gt_data)
    dv_mean_data_acc = accuracy(dv_mean_data, gt_data)
    dv_mean_data_com = completeness(dv_mean_data, gt_data)
    dv_mean_scores[0].append(dv_mean_data_rsme)
    dv_mean_scores[1].append(dv_mean_data_acc)
    dv_mean_scores[2].append(dv_mean_data_com)
    # import ipdb;ipdb.set_trace()    
    print('RMSE of deep vote with mean pooling = %.4f'%dv_mean_data_rsme)
    print('Accuracy of deep vote with mean pooling = %.4f'%dv_mean_data_acc)
    print('Completeness of deep vote with mean pooling = %.4f\n'%dv_mean_data_com)

    # deep vote with max pooling
    # dv_max_data = np.load(os.path.join(dv_max_path, filename+'.npy'))
    # if save_fig:
    #     plt.imshow(getColorMapFromPalette(dv_max_data, fire_palette))
    #     plt.axis('off')
    #     plt.savefig(out_path+'_dv_max.png')
    # dv_max_data_rsme = RSME(dv_max_data, gt_data)
    # dv_max_data_acc = accuracy(dv_max_data, gt_data)
    # dv_max_data_com = completeness(dv_max_data, gt_data)
    # dv_max_scores[0].append(dv_max_data_rsme)
    # dv_max_scores[1].append(dv_max_data_acc)
    # dv_max_scores[2].append(dv_max_data_com)
    # print('RMSE of deep vote with max pooling = %.4f'%dv_max_data_rsme)
    # print('Accuracy of deep vote with max pooling = %.4f'%dv_max_data_acc)
    # print('Completeness of deep vote with max pooling = %.4f\n'%dv_max_data_com)

print('Final RMSE of consense vote = %.4f'%np.mean(consense_scores[0]))
print('Final Accuracy of consense vote = %.4f'%np.mean(consense_scores[1]))
print('Final Completeness of consense vote = %.4f\n'%np.mean(consense_scores[2]))

# print('Final RMSE of median fusion = %.4f'%np.mean(median_scores[0]))
# print('Final Accuracy of median fusion = %.4f'%np.mean(median_scores[1]))
# print('Final Completeness of median fusion = %.4f\n'%np.mean(median_scores[2]))

# print('Final RMSE of kmedian_fusion vote = %.4f'%np.mean(kmedian_scores[0]))
# print('Final Accuracy of kmedian_fusion vote = %.4f'%np.mean(kmedian_scores[1]))
# print('Final Completeness of kmedian_fusion vote = %.4f\n'%np.mean(kmedian_scores[2]))

print('Final RMSE of deep vote = %.4f'%np.mean(dv_scores[0]))
print('Final Accuracy of deep vote = %.4f'%np.mean(dv_scores[1]))
print('Final Completeness of deep vote = %.4f\n'%np.mean(dv_scores[2]))

print('Final RMSE of deep vote with mean pooling = %.4f'%np.mean(dv_mean_scores[0]))
print('Final Accuracy of deep vote with mean pooling = %.4f'%np.mean(dv_mean_scores[1]))
print('Final Completeness of deep vote with mean pooling = %.4f\n'%np.mean(dv_mean_scores[2]))

# print('Final RMSE of deep vote with max pooling = %.4f'%np.mean(dv_max_scores[0]))
# print('Final Accuracy of deep vote with max pooling = %.4f'%np.mean(dv_max_scores[1]))
# print('Final Completeness of deep vote with max pooling = %.4f\n'%np.mean(dv_max_scores[2]))