import os
import time
import numpy as np

def save_height(outpath, hms, fns, mode):
    # hms: N x length x length
    for hm, fn in zip(hms, fns):
        np.save(os.path.join(os.path.join('../results', outpath, 'reconstruction_%s'%mode), fn+'.npy'), hm)

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