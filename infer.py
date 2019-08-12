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

from data_util import load_folder
from model_util import get_network

def infer_height2(args, gt_path, data_path):
    batch_size = args.batch_size
    AllX = []
    Ally = []
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    filenames = np.array(os.listdir(data_path))
    n_total = len(filenames)
    print('number of data: %d'%n_total)
    fold_size = n_total//args.n_folds
    shuffled_index = np.arange(n_total)
    np.random.seed(2019)
    np.random.shuffle(shuffled_index)
    test_ids = shuffled_index[list(range(0*fold_size, 1*fold_size))]
    n_test_batch = test_ids.shape[0]//batch_size
    filenames = filenames[test_ids]
    # N x P x W x H x 2
    for filename in filenames:
        file_tmp = np.load(os.path.join(data_path, filename, 'trainX.npy'))
        height_gt = np.load(os.path.join(gt_path, filename+'.npy'))
        AllX.append(file_tmp)
        Ally.append(height_gt)

    X_all, y_all = np.array(AllX), np.array(Ally)

    for i in range(4):
        
        for k in range(n_test_batch):
            X = Variable(torch.cuda.FloatTensor(X_all[k*batch_size:(k+1)*batch_size]), requires_grad=False)
            Y = Variable(torch.cuda.FloatTensor(y_all[k*batch_size:(k+1)*batch_size]), requires_grad=False)
            pred_height = D(X)
            output = pred_height.cpu().data.numpy()
            # import ipdb;ipdb.set_trace()
            for hm, fn in zip(output, filenames[k*batch_size:(k+1)*batch_size]):
                np.save(os.path.join(os.path.join(args.output, 'test_epochs'), fn+'_%d.npy'%(i*100+99)), hm)
            del X, Y, pred_height, output


def infer_height(D, x):
    # import ipdb;ipdb.set_trace()
    X = Variable(torch.cuda.FloatTensor(x).unsqueeze_(0), requires_grad=False)
    pred_height = D(X)
    output = pred_height.cpu().data.numpy()
    return output[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default='./cachedir', help='Save results filepath')
    parser.add_argument("-i", "--input", type=str, default='/home/songweig/LockheedMartin/data/MVS', help='input file folder')
    parser.add_argument("-ie", "--input_epoch", type=int, default=399, help='Load model after n epochs')
    parser.add_argument("-if", "--input_fold", type=int, default=4, help='Load model of fold n')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
    parser.add_argument('-np', '--n_pair', type=int, default=6, help='number of pair to be loaded')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('-n', '--exp_name', type=str, default='plain', help='the name to identify current experiment')
    parser.add_argument('-r', '--residual', type=bool, default=True, help='use residual learning or not')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    result_path = os.path.join(args.output, args.exp_name, 'reconstruction')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('loading the trained models...')
    D = get_network(args.model)(n_pairs=args.n_pair, residual=args.residual).cuda()
    D.load_state_dict(torch.load(os.path.join(args.output, args.exp_name, 'models', 'fold%d_%d'%(args.input_fold, args.input_epoch))))

    print('loading the data...')
    x = load_folder(args.input, n_pair=args.n_pair, mode='test')

    begin_time = time.time()
    print('infering the height...')
    x_fusion = infer_height(D, x)
    print('took %.4f s'%(time.time()-begin_time))

    target_file = os.path.join(result_path, os.path.basename(args.input)+'_fold%d_%d_pair%d.npy'%(args.input_fold, args.input_epoch, args.n_pair))
    print('saving results to %s...'%target_file)
    np.save(target_file, x_fusion)