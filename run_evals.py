import os
import argparse
import numpy as np
import imageio as misc
from shutil import copyfile

import utils.eval_util as eval_util

def partial_median(x):
    k = x.shape[0]//2
    med = np.median(x[:k][x[k:].astype(bool)])
    return med

def run(exp_name, mode, input_path, gt_path, save_fig, n_pair):
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    result_path = '../results/%s/evaluation_%s'%(exp_name, mode)
    dv_path = '../results/%s/reconstruction_%s'%(exp_name, mode)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    fire_palette = misc.imread(os.path.join('image/fire_palette.png'))[0][:, 0:3]
    error_palette = misc.imread('image/error_palette.jpg')[0][:, 0:3]

    consensus_scores = []
    median_fusion_scores = []
    mean_fusion_scores = []
    kmedian_fusion_scores = []
    dv_scores = []

    filenames = [fn[:-4] for fn in os.listdir(dv_path)]
    print('Number of test data: %d'%len(filenames))
    for filename in filenames:
        out_path = os.path.join(result_path, filename)
        print(filename)
        # load gt
        gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
            misc.imsave(out_path+'_gt_data.png', color_map)
        im_min, im_max = eval_util.getImageMinMax(gt_data)

        pair_data = []
        for i in range(n_pair):
            height = np.load(os.path.join(input_path, filename, input_temp%i))
            if save_fig:
                color_map = eval_util.getColorMapFromPalette(height, fire_palette, im_min, im_max)
                misc.imsave(out_path+'_input_height%d.png'%i, color_map)
                copyfile(os.path.join(input_path, filename, input_img_temp%i), out_path+'_input_color%d.png'%i)
            pair_data.append(height)
        pair_data = np.array(pair_data)

        # consensus vote
        con_data = np.load(os.path.join(input_path, filename, 'test.npz'))['f_infos'][:, :, 1]
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(con_data, fire_palette, im_min, im_max)
            misc.imsave(out_path+'_consensus.png', color_map)
        metrics = eval_util.evaluate(con_data, gt_data)
        consensus_scores.append(metrics)
        print('RMSE, Accuracy, Completeness, L1 Error of consensus vote: %.4f,  %.4f,  %.4f,  %.4f'%(
                metrics[0], metrics[1], metrics[2], metrics[3]))

        # mean fusion
        mean_data = np.mean(pair_data, 0)
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(mean_data, fire_palette, im_min, im_max)
            misc.imsave(out_path+'_mean.png', color_map)
        metrics = eval_util.evaluate(mean_data, gt_data)
        mean_fusion_scores.append(metrics)
        print('RMSE, Accuracy, Completeness, L1 Error of mean fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
                metrics[0], metrics[1], metrics[2], metrics[3]))

        # median fusion
        median_fusion = np.median(pair_data, 0)
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(median_fusion, fire_palette, im_min, im_max)
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
            color_map = eval_util.getColorMapFromPalette(kmedian_fusion, fire_palette, im_min, im_max)
            misc.imsave(out_path+'_kmedian_fusion.png', color_map)
        metrics = eval_util.evaluate(kmedian_fusion, gt_data)
        kmedian_fusion_scores.append(metrics)
        print('RMSE, Accuracy, Completeness, L1 Error of kmedian fusion: %.4f,  %.4f,  %.4f,  %.4f'%(
                metrics[0], metrics[1], metrics[2], metrics[3]))

        # deep vote
        dv_data = np.load(os.path.join(dv_path, filename+'.npy'))
        if save_fig:
            color_map = eval_util.getColorMapFromPalette(dv_data, fire_palette, im_min, im_max)
            misc.imsave(out_path+'_dv.png', color_map)
            color_map = eval_util.getColorMapFromPalette(np.abs(dv_data-gt_data), error_palette, 0, 10)
            misc.imsave(out_path+'_error.png', color_map)
        metrics = eval_util.evaluate(dv_data, gt_data)
        dv_scores.append(metrics)
        print('RMSE, Accuracy, Completeness, L1 Error of dv: %.4f,  %.4f,  %.4f,  %.4f'%(
                metrics[0], metrics[1], metrics[2], metrics[3]))

    print('Final RMSE, Accuracy, Completeness, L1 Error of consensus vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
        tuple(np.mean(consensus_scores, 0).tolist()+np.std(consensus_scores, 0).tolist()))
    print('Final RMSE, Accuracy, Completeness, L1 Error of mean fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
        tuple(np.mean(mean_fusion_scores, 0).tolist()+np.std(mean_fusion_scores, 0).tolist()))
    print('Final RMSE, Accuracy, Completeness, L1 Error of median fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
        tuple(np.mean(median_fusion_scores, 0).tolist()+np.std(median_fusion_scores, 0).tolist()))
    print('Final RMSE, Accuracy, Completeness, L1 Error of kmedian fusion: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
        tuple(np.mean(kmedian_fusion_scores, 0).tolist()+np.std(kmedian_fusion_scores, 0).tolist()))
    print('Final RMSE, Accuracy, Completeness, L1 Error of deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
        tuple(np.mean(dv_scores, 0).tolist()+np.std(dv_scores, 0).tolist()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='base', help='the name to identify current experiment')
    parser.add_argument('-np', '--n_pair', type=int, default=6, help='number of pair to be loaded')
    parser.add_argument('-m', '--mode', type=str, default='test', help='evaluate training or testing results')
    parser.add_argument('-s', '--save_image', type=bool, default=True, help='save the images or not')
    args = parser.parse_args()
    input_path = '../data/MVS'
    gt_path = '../data/DSM'
    run(args.exp_name, args.mode, input_path, gt_path, args.save_image, args.n_pair)