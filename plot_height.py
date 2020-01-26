import os
import argparse
import numpy as np
# from scipy import misc
import imageio as misc

import utils.eval_util as eval_util

def plot_diff(input_path, gt_path):
    result_path = os.path.join(os.path.dirname(input_path), os.path.basename(input_path).replace('npy', 'png'))

    fire_palette = misc.imread('image/fire_palette.png')[0][:, 0:3]
    error_palette = misc.imread('image/error_palette.jpg')[0][:, 0:3]

    dv_height = np.load(input_path)
    gt_height = np.load(gt_path)

    # import ipdb;ipdb.set_trace()
    print('saving results to %s...'%(os.path.dirname(result_path)))

    color_map = eval_util.getColorMapFromPalette(dv_height, fire_palette)
    misc.imsave(result_path.replace('.png', '_dv.png'), color_map)

    color_map = eval_util.getColorMapFromPalette(gt_height, fire_palette)
    misc.imsave(result_path.replace('.png', '_gt.png'), color_map)

    color_map = eval_util.getColorMapFromPalette(np.abs(dv_height-gt_height), error_palette, 0, 10)
    misc.imsave(result_path.replace('.png', '_error.png'), color_map)

    dv_height_l1e = eval_util.L1E(dv_height, gt_height)
    dv_height_rsme = eval_util.RSME(dv_height, gt_height)
    dv_height_acc = eval_util.accuracy(dv_height, gt_height)
    dv_height_com = eval_util.completeness(dv_height, gt_height)
    print('L1 Error of estimation = %.4f'%dv_height_l1e)
    print('RMSE of estimation = %.4f'%dv_height_rsme)
    print('Accuracy of estimation = %.4f'%dv_height_acc)
    print('Completeness of estimation = %.4f\n'%dv_height_com)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='../results/plain_res/reconstruction/dem_6_18_fold0_399_pair10.npy', help='input height map')
    parser.add_argument("-g", "--ground_truth", type=str, default='/home/songweig/LockheedMartin/data/DSM/dem_6_18.npy', help='ground truth height map')
    args = parser.parse_args()
    plot_diff(args.input, args.ground_truth)