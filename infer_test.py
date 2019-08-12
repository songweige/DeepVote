import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_util import load_folder
from model_util import get_network

def infer_test_height(model, out_path, data_path, fold_id, input_epoch=399):
    n_folds = 5
    n_pairs = 6
    residual = True
    batch_size = 5
    D = get_network(model)(n_pairs=n_pairs, residual=residual).cuda()
    AllX = []
    Ally = []
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'

    # load the test data
    filenames = np.array(os.listdir(data_path))
    n_total = len(filenames)
    print('number of data: %d'%n_total)
    fold_size = n_total//n_folds
    shuffled_index = np.arange(n_total)
    np.random.seed(2019)
    np.random.shuffle(shuffled_index)
    test_ids = shuffled_index[list(range(fold_id*fold_size, (fold_id+1)*fold_size))]
    n_test_batch = test_ids.shape[0]//batch_size
    filenames = filenames[test_ids]
    # N x P x W x H x 2
    for filename in filenames:
        file_tmp = np.load(os.path.join(data_path, filename, 'trainX.npy'))
        AllX.append(file_tmp)

    X_all = np.array(AllX)

    D.load_state_dict(torch.load(os.path.join(out_path, 'models', 'fold%d_%d'%(fold_id, input_epoch))))
    for k in range(n_test_batch+1):
        if k*batch_size == test_ids.shape[0]:
            continue
        if k == n_test_batch:
            X = Variable(torch.cuda.FloatTensor(X_all[k*batch_size:]), requires_grad=False)
        else:
            X = Variable(torch.cuda.FloatTensor(X_all[k*batch_size:(k+1)*batch_size]), requires_grad=False)
        pred_height = D(X)
        output = pred_height.cpu().data.numpy()
        # import ipdb;ipdb.set_trace()
        for hm, fn in zip(output, filenames[k*batch_size:(k+1)*batch_size]):
            np.save(os.path.join(os.path.join(out_path, 'reconstruction_test'), fn+'.npy'), hm)
        del X, pred_height, output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    data_path = '/home/songweig/LockheedMartin/data/MVS'

    # out_path = '../results/plain_res'
    # infer_test_height('base', out_path, data_path, fold_id=1)
    
    out_path = '../results/mean_res'
    infer_test_height('mean', out_path, data_path, fold_id=1)
    
    out_path = '../results/weighted'
    infer_test_height('weighted', out_path, data_path, fold_id=1)
    
    out_path = '../results/mean2'
    infer_test_height('mean2', out_path, data_path, fold_id=1)
    
    out_path = '../results/large'
    infer_test_height('large', out_path, data_path, fold_id=1)