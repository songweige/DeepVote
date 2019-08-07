import os
import cv2
import numpy as np
from scipy import misc

import torch
import torch.nn as nn
from torch.autograd import Variable

## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )
        

class MMMerge(nn.Module):
    """docstring for MMMerge"""
    def __init__(self):
        super(MMMerge, self).__init__()
        self.model = nn.Sequential(
            conv2d(batch_norm=False, in_planes=2, out_planes=8, kernel_size=5),
            conv2d(batch_norm=False, in_planes=8, out_planes=16, kernel_size=5),
            conv2d(batch_norm=False, in_planes=16, out_planes=8, kernel_size=5),
            conv2d(batch_norm=False, in_planes=8, out_planes=1, kernel_size=5), # predict the height for each pixel
        )
    def forward(self, img):
        return self.model(img)


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, X, y, i, n_pair=6, num_epochs=2000, batch_size=10):
        self.n_pair = n_pair
        self.n_epochs = num_epochs
        self.batch_size = batch_size
        self.D = MMMerge().cuda()
        # self.L = nn.MSELoss().cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1e-2)
        self.n_total = X.shape[0]
        self.shuffled_index = np.arange(self.n_total)
        np.random.shuffle(self.shuffled_index)
        self.inverse_index = np.argsort(self.shuffled_index)
        self.X = X[self.shuffled_index]
        self.y = y[self.shuffled_index]
        self.seg = i # interval
        self.seg_id = [0]
        for item in self.seg[1:]:
            self.seg_id.append(self.seg_id[-1]+item[0]*item[1])
        # rng_state = np.random.get_state()
        # np.random.shuffle(self.X)
        # np.random.set_state(rng_state)
        # np.random.shuffle(self.y)
        self.mask = np.logical_not(self.y<0).astype(float)
        print(self.L(Variable(torch.cuda.FloatTensor(self.X[:, 0, 1, :, :]*self.mask)), Variable(torch.cuda.FloatTensor(self.y*self.mask))))
        self.filenames = ['out_MasterProvisional1', 'out_MasterProvisional2', 'out_MasterProvisional3', 'out_MasterSequestered1', 
                    'out_MasterSequestered2', 'out_MasterSequestered3', 'out_MasterSequesteredPark']

        self.width, self.height = X[0][0].shape[1:]

    def weights_init(self, m):
        try:
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        except:
            m.data.normal_(0, 0.01)

    def save_height(self, hm, margins, fn):
        # hm: N x length x length
        print(hm.shape, margins)
        assert(hm.shape[1]==hm.shape[2])
        output = np.zeros([margins[0]*100, margins[1]*100], dtype=np.float64)
        for j in range(margins[0]):
            for k in range(margins[1]):
                output[j*100:(j+1)*100, k*100:(k+1)*100] = hm[j*margins[1]+k]
        np.save(fn+'.npy', output)


    def run(self):
        self.D.train()
        loss_val = float('inf')
        self.train_loss = []
        outputs = []
        cv_losses = []
        for i in range(7):
            test_ids = self.inverse_index[self.seg_id[i]:self.seg_id[i+1]]
            train_ids = np.setdiff1d(np.arange(self.n_total), test_ids)
            # import ipdb;ipdb.set_trace()
            Xtrain = self.X[train_ids]
            Xtest = self.X[test_ids]
            ytrain = self.y[train_ids]
            ytest = self.y[test_ids]
            masktrain = self.mask[train_ids]
            masktest = self.mask[test_ids]
            n_batch = Xtrain.shape[0]//self.batch_size
            for p in self.D.parameters():
                self.weights_init(p)
            for j in range(self.n_epochs):
                train_epoch_loss = []
                for k in range(n_batch):
                    #forward calculation and back propagation, X: B x 6 x 2x x W x H
                    X = Variable(torch.cuda.FloatTensor(Xtrain[k*self.batch_size:(k+1)*self.batch_size]).view(-1, 2, self.width, self.height))
                    Y = Variable(torch.cuda.FloatTensor(ytrain[k*self.batch_size:(k+1)*self.batch_size]))
                    Mask = Variable(torch.cuda.FloatTensor(masktrain[k*self.batch_size:(k+1)*self.batch_size]))
                    # if j == 100:
                    #     import ipdb; ipdb.set_trace()
                    self.optimizer.zero_grad()
                    pred_height = self.D(X).view(-1, self.n_pair, self.width, self.height)
                    pred_height = pred_height.mean(1)
                    # print(pred_height.shape, Y.shape)
                    loss = self.L(pred_height*Mask, Y*Mask)
                    loss_val = loss.data.cpu().numpy()
                    loss.backward()
                    self.optimizer.step()
                    train_epoch_loss.append(loss_val)
                    del X,Y,pred_height,loss
                
                # test
                X = Variable(torch.cuda.FloatTensor(Xtest).view(-1, 2, self.width, self.height))
                Y = Variable(torch.cuda.FloatTensor(ytest))
                Mask = Variable(torch.cuda.FloatTensor(masktest))
                pred_height = self.D(X).view(-1, self.n_pair, self.width, self.height)
                pred_height = pred_height.mean(1)
                loss = self.L(pred_height*Mask, Y*Mask)
                loss_val = loss.data.cpu().numpy()
                print("Fold %d, Epochs %d, training loss: %f, test loss %f"%(i, j, np.mean(train_epoch_loss), loss_val))
            output = (pred_height.cpu().data.numpy())
            self.save_height(output, self.seg[i+1], '../results/output/%s/%s'%(self.filenames[i], self.filenames[i]+'l1'))
            cv_losses.append(loss_val)
            print("Fold %d, test loss %f"%(i, loss_val))
        print('overall performance: %f'%np.mean(cv_losses))
        return cv_losses

# load input sets and ground truth
def load_data(result_path, data_path, n_pair=6):
    file2range = {'out_MasterProvisional1': [214, 736, 211, 747], 'out_MasterProvisional2': [176, 794, 183, 824], 
                'out_MasterProvisional3': [167, 614, 208, 778], 'out_MasterSequestered1': [218, 701, 205, 701], 
                'out_MasterSequestered2': [218, 679, 203, 745], 'out_MasterSequestered3': [205, 708, 212, 626], 
                'out_MasterSequesteredPark': [229, 906, 217, 985]}
    # load input data
    AllX = []
    Ally = []
    Allimage = [[0, 0]]
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    # N x 6 x W x H x 2
    for key in sorted(file2range.keys()):
        x1, x2, y1, y2 = file2range[key]
        x_margin = (x2-x1)//100
        y_margin = (y2-y1)//100
        n_prev = len(AllX)
        Allimage.append([y_margin, x_margin])
        for _ in range(y_margin):
            for _ in range(x_margin):
                AllX.append([])
                Ally.append([])
        for i in range(n_pair):
            height = np.load(os.path.join(data_path, key, 'tmp', input_temp%i))[y1:y2, x1:x2]
            img = misc.imread(os.path.join(data_path, key, 'tmp', input_img_temp%i))[y1:y2, x1:x2]
            # crop into 100 x 100 pieces
            for j in range(y_margin):
                for k in range(x_margin):
                    AllX[n_prev+j*x_margin+k].append(np.dstack((img[j*100:(j+1)*100, k*100:(k+1)*100], height[j*100:(j+1)*100, k*100:(k+1)*100])).transpose(2, 0, 1))
        # load gt
        height_gt = np.rot90(misc.imread(os.path.join(result_path, key[4:]+'.tif')), -1)
        if 'Provisional' in key:
            print(key)
            mask_ids = np.where(height_gt>0)
            x1_corr, y1_corr, x2_corr, y2_corr = np.min(mask_ids[1]), np.min(mask_ids[0]), np.max(mask_ids[1]), np.max(mask_ids[0])
            height_gt = height_gt[y1_corr:y2_corr, x1_corr:x2_corr]

        height_gt = cv2.resize(height_gt, (x2-x1, y2-y1))
        for j in range(y_margin):
            for k in range(x_margin):
                Ally[n_prev+j*x_margin+k] = height_gt[j*100:(j+1)*100, k*100:(k+1)*100]


    AllX, Ally = np.array(AllX), np.array(Ally)
    mask = np.isnan(AllX)
    AllX[mask] = 0
    print(AllX.shape, Ally.shape, Allimage)
    return AllX, Ally, Allimage


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    input_path = '../data/'
    n_pair = 6
    X, y, i = load_data(input_path, input_path, n_pair)
    trainer = Trainer(X, y, i)
    trainer.run()
