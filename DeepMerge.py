import os
import cv2
import time
import argparse
import numpy as np
from scipy import misc

from model_util import *

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args, X, y, filenames):
        self.args = args
        self.n_pair = args.n_pair
        self.n_folds = args.n_folds
        self.n_epochs = args.epochs
        self.batch_size = args.batch_size
        if args.model == 'plain':
            self.D = MMMerge(self.n_pair, args.residual).cuda()
        elif args.model == 'mean':
            self.D = MMMerge_mean(self.n_pair, args.residual).cuda()
        elif args.model == 'mean2':
            self.D = MMMerge_mean2(self.n_pair, args.residual).cuda()
        elif args.model == 'median':
            self.D = MMMerge_median(self.n_pair, args.residual).cuda()
        elif args.model == 'max':
            self.D = MMMerge_max(self.n_pair, args.residual).cuda()
        else:
            print("Wrong model")
        # self.L = nn.MSELoss().cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1.)
        self.n_total = X.shape[0]
        self.shuffled_index = np.arange(self.n_total)
        np.random.seed(2019)
        np.random.shuffle(self.shuffled_index)
        self.X = X
        self.y = y
        self.filenames = filenames
        self.width, self.height = X[0][0].shape[1:]

    def weights_init(self, m):
        try:
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        except:
            m.data.normal_(0, 0.01)

    def save_height(self, hms, fns, mode):
        # hms: N x length x length
        for hm, fn in zip(hms, fns):
            np.save(os.path.join(os.path.join('../results', self.args.output, 'reconstruction_%s'%mode), fn+'.npy'), hm)

    def run(self):
        self.D.train()
        loss_val = float('inf')
        self.train_loss = []
        outputs = []
        cv_losses = []
        fold_size = self.n_total//self.n_folds
        # 5 cross validation
        print('start training')
        for i in range(1):
        # for i in range(self.n_folds):
            test_ids = self.shuffled_index[list(range(i*fold_size, (i+1)*fold_size))]
            train_ids = np.setdiff1d(self.shuffled_index, test_ids)
            # import ipdb;ipdb.set_trace()
            n_batch = train_ids.shape[0]//self.batch_size
            n_test_batch = test_ids.shape[0]//self.batch_size
            for p in self.D.parameters():
                self.weights_init(p)
            for j in range(self.n_epochs):
                np.random.shuffle(train_ids)
                begin = time.time()
                train_epoch_loss = []
                test_epoch_loss = []
                # import ipdb;ipdb.set_trace()
                for k in range(n_batch+1):
                    #forward calculation and back propagation, X: B x P x 2 x W x H
                    # import ipdb;ipdb.set_trace()
                    if k == n_batch:
                        train_batch_ids = train_ids[k*self.batch_size:]
                    else:
                        train_batch_ids = train_ids[k*self.batch_size:(k+1)*self.batch_size]
                    X = Variable(torch.cuda.FloatTensor(self.X[train_batch_ids]), requires_grad=False)
                    Y = Variable(torch.cuda.FloatTensor(self.y[train_batch_ids]), requires_grad=False)
                    self.optimizer.zero_grad()
                    pred_height = self.D(X)
                    loss = self.L(pred_height, Y)
                    loss_val = loss.data.cpu().numpy()
                    loss.backward()
                    self.optimizer.step()
                    train_epoch_loss.append(loss_val)
                    # save the last training estimation
                    if j == self.n_epochs-1:
                        output = (pred_height.cpu().data.numpy())
                        self.save_height(output, filenames[train_batch_ids], 'train')
                    del X,Y,pred_height,loss
                
                if (j+1)%100 == 0:
                    torch.save(self.D.state_dict(), os.path.join('../results', self.args.output, 'models', 'fold%d_%d'%(i, j)))
                print("Fold %d, Epochs %d, time = %ds, training loss: %f"%(i, j, time.time() - begin, np.mean(train_epoch_loss)))
                            # test
            for k in range(n_test_batch+1):
                if k == n_test_batch:
                    test_batch_ids = test_ids[k*self.batch_size:]
                else:
                    test_batch_ids = test_ids[k*self.batch_size:(k+1)*self.batch_size]
                X = Variable(torch.cuda.FloatTensor(self.X[test_batch_ids]), requires_grad=False)
                Y = Variable(torch.cuda.FloatTensor(self.y[test_batch_ids]), requires_grad=False)
                pred_height = self.D(X)
                loss = self.L(pred_height, Y)
                loss_val = loss.data.cpu().numpy()
                test_epoch_loss.append(loss_val)
                output = (pred_height.cpu().data.numpy())
                self.save_height(output, filenames[test_batch_ids], 'test')
                del X,Y,pred_height,loss
            print("Fold %d, Epochs %d, time = %ds, training loss: %f, test loss %f"%(i, j, time.time() - begin, np.mean(train_epoch_loss), np.mean(test_epoch_loss)))

            cv_losses.append(np.mean(test_epoch_loss))
        print('overall performance: %f'%np.mean(cv_losses))
        return cv_losses

# load input sets and ground truth
def load_data(gt_path, data_path, n_pair=6):
    # load input data
    AllX = []
    Ally = []
    input_temp = 'FF-%d.npy'
    input_img_temp = '%d-color.png'
    filenames = os.listdir(data_path)
    # N x P x W x H x 2
    begin_time = time.time()
    print('start to load data')
    for filename in filenames:
        # file_tmp = []
        # for i in range(n_pair):
        #     height = np.load(os.path.join(data_path, filename, input_temp%i))
        #     img = misc.imread(os.path.join(data_path, filename, input_img_temp%i))
        #     file_tmp.append(np.dstack((img, height)).transpose(2, 0, 1))
        # load gt
        file_tmp = np.load(os.path.join(data_path, filename, 'trainX.npy'))
        height_gt = np.load(os.path.join(gt_path, filename+'.npy'))
        AllX.append(file_tmp)
        Ally.append(height_gt)
    AllX, Ally, filenames = np.array(AllX), np.array(Ally), np.array(filenames)
    print('it took %.2fs to load data'%(time.time()-begin_time))
    return AllX, Ally, filenames


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
    parser.add_argument('-r', '--residual', type=bool, default=False, help='use residual learning or not')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('../results', args.output, 'models')):
        os.mkdir(os.path.join('../results', args.output))
        os.mkdir(os.path.join('../results', args.output, 'models'))
        os.mkdir(os.path.join('../results', args.output, 'reconstruction_train'))
        os.mkdir(os.path.join('../results', args.output, 'reconstruction_test'))

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_path = '/home/songweig/LockheedMartin/data/MVS'
    gt_path = '/home/songweig/LockheedMartin/data/DSM'
    X, y, filenames = load_data(gt_path, data_path, args.n_pair)
    # import ipdb;ipdb.set_trace()
    X[:, :, 0, :, :] = X[:, :, 0, :, :]/255.
    trainer = Trainer(args, X, y, filenames)
    if args.load_model:
        trainer.D.load_state_dict(torch.load(os.path.join(args.output, 'models', '%s_%d'%(args.input, args.input_epoch))))
    trainer.run()
