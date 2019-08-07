import os
import time
import argparse
import numpy as np
from scipy import misc

import torch
import torch.nn as nn
from torch.autograd import Variable

## 2D convolution layers
class conv2d(nn.Module):
    def __init__(self, n_pairs, batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
        super(conv2d, self).__init__()
        if batch_norm:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.2,inplace=True),
            )
        else:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                nn.LeakyReLU(0.2,inplace=True),
            )
        self.n_pairs = n_pairs
    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x_conv = self.conv_layer(x)
        # print(x_conv.shape)
        x_conv = x_conv.view(-1, self.n_pairs, x_conv.size(1), x_conv.size(2), x_conv.size(3))
        return x_conv

class PermEqui_mean(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_mean, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.mean(1, keepdim=True).repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x

class PermEqui_median(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_median, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.median(1, keepdim=True)[0].repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x

class PermEqui_max(nn.Module):
  def __init__(self, n_pairs, in_dim, out_dim):
    super(PermEqui_max, self).__init__()
    self.n_pairs = n_pairs
    self.Gamma = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)
    # self.Lambda = conv2d(n_pairs, batch_norm=False, in_planes=in_dim, out_planes=out_dim, kernel_size=5)

  def forward(self, x):
    xm = x.max(1, keepdim=True)[0].repeat(1, self.n_pairs, 1, 1, 1)
    # xm = self.Lambda.forward(xm)
    # x = self.Gamma.forward(x)
    x = self.Gamma.forward(x-xm)
    # x = x - xm
    return x



class MMMerge_mean(nn.Module):
    """docstring for MMMerge_mean"""
    def __init__(self, n_pairs):
        super(MMMerge_mean, self).__init__()
        self.model = nn.Sequential(
            PermEqui_mean(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_mean(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_mean(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_mean(n_pairs, in_dim=8, out_dim=1,), 
        )
    def forward(self, img):
        return self.model(img).squeeze_().mean(1)

class MMMerge_median(nn.Module):
    """docstring for MMMerge_median"""
    def __init__(self, n_pairs):
        super(MMMerge_median, self).__init__()
        self.model = nn.Sequential(
            PermEqui_median(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_median(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_median(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_median(n_pairs, in_dim=8, out_dim=1,), 
        )
    def forward(self, img):
        return self.model(img).squeeze_().median(1)[0]

class MMMerge_max(nn.Module):
    """docstring for MMMerge_max"""
    def __init__(self, n_pairs):
        super(MMMerge_max, self).__init__()
        self.model = nn.Sequential(
            PermEqui_max(n_pairs, in_dim=2, out_dim=8,),
            PermEqui_max(n_pairs, in_dim=8, out_dim=16,),
            PermEqui_max(n_pairs, in_dim=16, out_dim=8,),
            PermEqui_max(n_pairs, in_dim=8, out_dim=1,), 
        )
    def forward(self, img):
        return self.model(img).squeeze_().max(1)[0]

class MMMerge(nn.Module):
    """docstring for MMMerge"""
    def __init__(self, n_pairs):
        super(MMMerge, self).__init__()
        self.model = nn.Sequential(
            conv2d(n_pairs, batch_norm=False, in_planes=2, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=16, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=16, out_planes=8, kernel_size=5),
            conv2d(n_pairs, batch_norm=False, in_planes=8, out_planes=1, kernel_size=5), # predict the height for each pixel
        )
    def forward(self, img):
        return self.model(img).squeeze_().mean(1)

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
    evaluate(args, gt_path, data_path)