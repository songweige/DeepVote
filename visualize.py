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


def medieanerror(hm_input, hm_gt):
    hm_gt = cv2.resize(hm_gt, (hm_input.shape[1], hm_input.shape[0]))
    print(hm_input.shape, hm_gt.shape)
    width, height = hm_input.shape
    not_nan_mask = np.logical_and(np.logical_not(np.isnan(hm_input)), np.logical_not(hm_gt<0))
    error = np.median(np.abs(hm_gt-hm_input)[not_nan_mask])
    return error, hm_gt


result_path = '../data/'
data_path = '../data/'

fire_palette = misc.imread(os.path.join(data_path, 'fire_palette.png'))[0][:, 0:3]
confidence_palette = misc.imread(os.path.join(data_path, 'confidence_palette.png'))[0][:, 0:3]

# rotate 180 degree
gt_data = np.rot90(misc.imread(os.path.join(data_path, 'Explorer.tif')), -1)
# plt.imshow(getColorMapFromPalette(gt_data, fire_palette))
# plt.show()


cand_data = np.load(os.path.join(result_path, 'FF-0.npy'))
# plt.imshow(getColorMapFromPalette(cand_data[y1:y2, x1:x2], fire_palette))
# plt.show()

# find the nan mask, x1, y1, x2, x2 1->min 2->max
mask_ids = np.where(np.logical_not(np.isnan(cand_data)))
x1, y1, x2, x2 = np.min(mask_ids[1]), np.min(mask_ids[0]), np.max(mask_ids[1]), np.max(mask_ids[0])

medieanerror(cand_data, gt_data)

x1 = 140
x2 = 1240
y1 = 148
y2 = 1331

error_min = 100.0

for x1d in range(50):
    for x2d in range(50):
        for y1d in range(50):
            for y2d in range(50): 
                error_now = medieanerror(cand_data[y1+y1d:y2-y2d, x1+x1d:x2-x2d], gt_data)
                if error_now < error_min:
                    print(y1+y1d, y2-y2d, x1+x1d, x2-x2d, error_now)
                    error_min = error_now
                    params = [y1+y1d, y2-y2d, x1+x1d, x2-x2d]


x1 = 175
x2 = 1240
y1 = 150
y2 = 1331

plt.imshow(getColorMapFromPalette(cand_data[y1:y2, x1:x2], fire_palette))
plt.savefig('input.png')
e, gt_data_resized = medieanerror(cand_data[y1:y2, x1:x2], gt_data)
plt.imshow(getColorMapFromPalette(gt_data_resized, fire_palette))
plt.savefig('gt.png')

error_map = cand_data[y1:y2, x1:x2]-gt_data_resized
error_map[gt_data_resized<0] = 0
error_palette = np.zeros((255, 3))
error_palette[:, 0] = np.arange(255)/255.
plt.imshow(getColorMapFromPalette(np.abs(error_map), error_palette))
plt.show()