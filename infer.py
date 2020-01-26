"""
Given n pairs of uncalibrated height maps and greyscale images, predict the calibrated height using the trained DeepVote model
"""
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.data_util import load_folder
from utils.model_util import get_network


def infer_height(D, x):
    X = Variable(torch.cuda.FloatTensor(x).unsqueeze_(0), requires_grad=False)
    pred_height = D(X)
    output = pred_height.cpu().data.numpy()
    return output[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default='/home/songweig/LockheedMartin/data/MVS', help='input file folder')
    parser.add_argument("-mn", "--model_name", type=str, default='fold0_399', help='the name of pretrained model')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
    parser.add_argument('-np', '--n_pair', type=int, default=6, help='number of pair to be loaded')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='base', help='which model to be used')
    parser.add_argument('-n', '--exp_name', type=str, default='base', help='the name to identify current experiment')
    parser.add_argument('-r', '--residual', type=bool, default=True, help='use residual learning or not')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    result_path = os.path.join('../results', args.exp_name, 'reconstruction')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('loading the trained models...')
    D = get_network(args.model)(n_pairs=args.n_pair, residual=args.residual).cuda()
    print(os.path.join('../results', args.exp_name, 'models', args.model_name))
    D.load_state_dict(torch.load(os.path.join('../results', args.exp_name, 'models', args.model_name)))

    print('loading the data...')
    x = load_folder(args.input, n_pair=args.n_pair, mode='test')

    begin_time = time.time()
    print('infering the height...')
    x_fusion = infer_height(D, x)
    print('took %.4f s'%(time.time()-begin_time))

    target_file = os.path.join(result_path, os.path.basename(args.input)+'_%s_%s.npy'%(args.exp_name, args.model_name))
    print('saving results to %s...'%target_file)
    np.save(target_file, x_fusion)