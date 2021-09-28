from nilearn.image import load_img
import numpy as np


fmap_img = load_img('../../../results/Cat12/spmF_0002.nii')

img_data = fmap_img.get_data()
max_F = np.argmax(img_data,  axis=None)
ind = np.unravel_index(max_F, img_data.shape)
print("Max F value found for voxel {}".format(ind))
debug = True