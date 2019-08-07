import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def getImageMinMax(im_np):
    return np.nanpercentile(im_np, 10), np.nanpercentile(im_np, 90)


def getColorMapFromPalette(im_np, palette, im_min = None, im_max = None):
    if (im_min is None):
        im_min, im_max = getImageMinMax(im_np)
    print('min:', im_min, 'max:', im_max)
    normalized = ((im_np - im_min) / (im_max - im_min))
    normalized[normalized < 0] = 0
    normalized[normalized > 1] = 1
    normalized_nan = np.isnan(normalized)
    normalized_not_nan = np.logical_not(normalized_nan)
    color_indexes = np.round(normalized * (palette.shape[0] - 2) + 1).astype('uint32')
    color_indexes[normalized_nan] = 0
    color_map = palette[color_indexes]
    return color_map

def RSME(inh, gth):
    input_h = np.copy(inh)
    gt_h = np.copy(gth)
    mask = np.logical_or(np.isnan(input_h), gt_h<0)
    count = mask.shape[0]*mask.shape[1]-np.sum(mask)
    input_h[mask] = 0.
    gt_h[mask] = 0.
    print('undefined: ', np.sum(mask), 'count: ', count)
    # return np.sum((input_h-gt_h)**2)
    return np.sqrt(np.sum((input_h-gt_h)**2)/count)

result_path = '../data/'
data_path = '../data/'

fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]


x1 = 175
x2 = 1240
y1 = 150
y2 = 1331

# rotate 180 degree
gt_data = np.rot90(misc.imread(os.path.join(data_path, 'Explorer.tif')), -1)
gt_data = cv2.resize(gt_data, (x2-x1, y2-y1))
plt.imshow(getColorMapFromPalette(gt_data, fire_palette))
plt.axis('off')
plt.savefig('../results/gt.png')

for i in range(6):
    cand_data = np.load(os.path.join(result_path, 'FF-%d.npy'%i))
    plt.imshow(getColorMapFromPalette(cand_data, fire_palette))
    plt.axis('off')
    plt.savefig('../results/FF-%d.png'%i)
print('Error of cand vote = %.4f'%RSME(cand_data[y1:y2, x1:x2], gt_data))


# consense vote
con_data = np.load(os.path.join(result_path, 'test_2.npz'))['f_infos'][:, :, 1]
x1_con = 45
x2_con = 1110
y1_con = 20
y2_con = 1201
plt.imshow(getColorMapFromPalette(con_data[y1_con:y2_con, x1_con:x2_con], fire_palette))
plt.axis('off')
plt.savefig('../results/consense.png')
print('Error of consense vote = %.4f'%RSME(con_data[y1_con:y2_con, x1_con:x2_con], gt_data))


# median fusion
pair_data = []
for i in range(6):
    pair_data.append(np.load(os.path.join(result_path, 'FF-%d.npy'%i)))


pair_data = np.array(pair_data)
median_fusion = np.zeros((y2-y1, x2-x1))
for i in range(y1, y2):
    for j in range(x1, x2):
        median_fusion[i-y1, j-x1] = np.median(pair_data[:, i, j])


plt.imshow(getColorMapFromPalette(median_fusion, fire_palette))
plt.axis('off')
plt.savefig('../results/median_fusion.png')
print('Error of median fusion = %.4f'%RSME(median_fusion, gt_data))


# deep vote
deep_data = np.load(os.path.join(result_path, 'output.npy'))
plt.imshow(getColorMapFromPalette(deep_data, fire_palette))
plt.axis('off')
plt.savefig('../results/deepvote.png')
print('Error of deepvote vote = %.4f'%RSME(deep_data, gt_data[:1000, :1000]))


# k-medians
pair_data = np.array(pair_data)
kmedian_fusion = np.zeros((y2-y1, x2-x1))
for i in range(y1, y2):
    for j in range(x1, x2):
        heights = pair_data[:, i, j]
        mask = np.logical_not(np.isnan(heights))
        if sum(mask) == 0:
            kmedian_fusion[i-y1, j-x1] = np.nan
            continue
        hmax, hmin = np.max(heights[mask]), np.min(heights[mask])
        max_set = []
        min_set = []
        for item in heights[mask]:
            max_dis, min_dis = hmax-item, hmin-item
            if max_dis>min_dis:
                min_set.append(item)
            else:
                max_set.append(item)
        if len(max_set) == 0 or len(min_set) == 0:
            kmedian_fusion[i-y1, j-x1] = np.median(pair_data[:, i, j])
        else:
            dis = np.median(max_set)-np.median(min_set)
            if dis > 5.:
                kmedian_fusion[i-y1, j-x1] = np.median(max_set[:, i, j])
            else:
                kmedian_fusion[i-y1, j-x1] = np.median(pair_data[:, i, j])
plt.imshow(getColorMapFromPalette(kmedian_fusion, fire_palette))
plt.axis('off')
plt.savefig('../results/kmedian_fusion.png')
print('Error of kmedian_fusion vote = %.4f'%RSME(kmedian_fusion, gt_data))