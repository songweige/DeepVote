import os
import time
import argparse
import numpy as np
from scipy import misc

import torch
import torch.nn as nn
from torch.autograd import Variable

from model_util import *

# load input sets and ground truth
def load_data(gt_path, data_path, n_pair=6):
    # load input data
    AllX = []
    Ally = []
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    filenames = os.listdir(data_path)
    # N x P x W x H x 2
    for filename in filenames:
        file_tmp = np.load(os.path.join(data_path, filename, 'trainX.npy'))
        height_gt = np.load(os.path.join(gt_path, filename+'.npy'))
        AllX.append(file_tmp)
        Ally.append(height_gt)
    AllX, Ally, filenames = np.array(AllX), np.array(Ally), np.array(filenames)
    mask = np.isnan(AllX)
    AllX[mask] = 0
    return AllX, Ally, filenames

def evaluate_epoch(args, gt_path, data_path):
    batch_size = args.batch_size
    D = MMMerge(args.n_pair).cuda()
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
        D.load_state_dict(torch.load(os.path.join(args.output, 'models', 'fold0_%d'%(i*100+99))))
        for k in range(n_test_batch):
            X = Variable(torch.cuda.FloatTensor(X_all[k*batch_size:(k+1)*batch_size]), requires_grad=False)
            Y = Variable(torch.cuda.FloatTensor(y_all[k*batch_size:(k+1)*batch_size]), requires_grad=False)
            pred_height = D(X)
            output = pred_height.cpu().data.numpy()
            # import ipdb;ipdb.set_trace()
            for hm, fn in zip(output, filenames[k*batch_size:(k+1)*batch_size]):
                np.save(os.path.join(os.path.join(args.output, 'test_epochs'), fn+'_%d.npy'%(i*100+99)), hm)
            del X, Y, pred_height, output

def evaluate(args, gt_path, data_path):
    D = MMMerge(args.n_pair).cuda()
    X_all, y_all, filenames = load_data(gt_path, data_path, args.n_pair)
    n_total = X_all.shape[0]
    batch_size = args.batch_size
    fold_size = n_total//args.n_folds
    shuffled_index = np.arange(n_total)
    np.random.seed(2019)
    np.random.shuffle(shuffled_index)
    for i in range(4, 5):
        test_ids = shuffled_index[list(range(i*fold_size, (i+1)*fold_size))]
        train_ids = np.setdiff1d(shuffled_index, test_ids)
        n_batch = train_ids.shape[0]//batch_size
        n_test_batch = test_ids.shape[0]//batch_size
        print('fold%d'%i)
        D.load_state_dict(torch.load(os.path.join(args.output, 'models', 'fold%d_300'%i)))
        for k in range(n_batch):
            # test_batch_ids = test_ids[k*batch_size:(k+1)*batch_size]
            # X = Variable(torch.cuda.FloatTensor(X_all[test_batch_ids]), requires_grad=False)
            # Y = Variable(torch.cuda.FloatTensor(y_all[test_batch_ids]), requires_grad=False)
            train_batch_ids = train_ids[k*batch_size:(k+1)*batch_size]
            X = Variable(torch.cuda.FloatTensor(X_all[train_batch_ids]), requires_grad=False)
            Y = Variable(torch.cuda.FloatTensor(y_all[train_batch_ids]), requires_grad=False)
            pred_height = D(X)
            output = pred_height.cpu().data.numpy()
            # import ipdb;ipdb.set_trace()
            for hm, fn in zip(output, filenames[train_batch_ids]):
                np.save(os.path.join(os.path.join(args.output, 'reconstruction_train'), fn+'.npy'), hm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default='../results', help='Save results filepath')
    parser.add_argument("-ie", "--input_epoch", type=str, default=None, help='Load model after n epochs')
    parser.add_argument("-i", "--input", type=str, default='fold0', help='Load model filepath')
    parser.add_argument("-ld", "--load_model", type=bool, default=False, help='Load pretrained model or not')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-np', '--n_pair', type=int, default=6, help='number of pair to be loaded')
    parser.add_argument('-nf', '--n_folds', type=int, default=5, help='number of pair to be loaded')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output, 'models')):
        os.mkdir(os.path.join(args.output, 'models'))
        os.mkdir(os.path.join(args.output, 'reconstruction_train'))

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_path = '/home/songweig/LockheedMartin/data/MVS'
    gt_path = '/home/songweig/LockheedMartin/data/DSM'
    evaluate_epoch(args, gt_path, data_path)
    # evaluate(args, gt_path, data_path)