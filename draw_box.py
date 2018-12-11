from __future__ import print_function
import cv2
import numpy as np
import glob
import os
from scipy.stats import multivariate_normal
from scipy.io import loadmat
import matplotlib.pyplot as plt


'''
draw box on object (vehicle)
for each object, the format of return: (x,y,w,h)
# index the index range of image to be labelled
'''
def img_labelling(img_path, index):
    fromCenter = False
    showCrosshair = False
    img_list = glob.glob(img_path)
    label_list = []
    idx_start = int(index.split('-')[0]) - 1
    idx_end = int(index.split('-')[1])
    for i, img_path in enumerate(img_list):
        print(i)
        if i < idx_start:
            continue
        if i in range(idx_start, idx_end):
            img = cv2.imread(img_path)
            label = cv2.selectROIs(img_path, img, fromCenter, showCrosshair)
            if len(label) == 0:
                label_list.append(['-'])
            else:
                label_list.append([label])
            cv2.destroyAllWindows()
        else:
            break
    return label_list


'''
crop image first by general region of interest
'''
# img_shape: [1080, 1920] (h, w)
# ratios = [1/4, 1/3, 8/9]
def resize_imgs(img_paths, output_path, ratios, img_shape):
    img_list = glob.glob(img_paths)
    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        crop_img = img[int(np.floor(img_shape[0]*ratios[0])):img_shape[0],
                       int(np.floor(img_shape[1]*ratios[1])):int(np.floor(img_shape[1]*ratios[2]))]
        # save to folder
        cv2.imwrite(output_path+'/'+img_path.split('\\')[1], crop_img)
        print(i)
    return 0


'''
crop image first by general region of interest, for the second camera
'''
# img_shape: [1080, 1920] (h, w)
# ratios = [0.8/6, 4.0/6, 1.0/5, 3.5/5]
def resize_imgs_v2(img_paths, output_path, ratios, img_shape):
    img_list = glob.glob(img_paths)
    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        crop_img = img[int(np.floor(img_shape[0]*ratios[0])):int(np.floor(img_shape[0]*ratios[1])),
                       int(np.floor(img_shape[1]*ratios[2])):int(np.floor(img_shape[1]*ratios[3]))]
        # save to folder
        cv2.imwrite(output_path+'/'+img_path.split('\\')[1], crop_img)
        print(i)
    return 0


'''
- Objective:
    - get rotated covariance matrix
- Input:
    - w, h: length of the bounding box along y, x axis in standard coordinate
- Notice:
    - tunning parameter: the variance on x, y axis could be tuned
'''
def rotateCovarianceMatrix(w, h):
    sigma_x = np.float(w)/6
    sigma_y = np.float(h)/6
    cov_mat_org = np.asarray([[np.square(sigma_x), 0], [0, np.square(sigma_y)]])
    sin_v = np.float(h**2)/(w**2+h**2)
    cos_v = -np.float(w**2)/(w**2+h**2)
    covert_mat = np.asarray([[cos_v, sin_v], [-sin_v, cos_v]])
    return np.matmul(np.matmul(covert_mat, cov_mat_org), covert_mat.T)


'''
- Input:
    - img_dim: [height, width]
    - loc: location of the bounding box [x,y,w,h]
'''
def generateSingle2DGaussian(img_dim, loc):
    x, y, w, h = loc
    H, W = img_dim
    Y, X = np.mgrid[0:H:1, 0:W:1]
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # get rotated covariance matrix
    cov_m = rotateCovarianceMatrix(w, h)
    rv = multivariate_normal([x + w/2, y + h/2], cov_m)
    result = rv.pdf(pos)
    return result


'''
- Objective:
    - generate ground truth density (with resize)
- Steps: 
    - 1) simulate rotated 2D Gaussian, then multiply with mask map
    - 2) resize to 224 * 224? 
- Input:
    mask: [height, width] (810. 1066)
    input_list:
    resize_shape: tuple (224, 224)
'''
def generateDensityMap(mask, input_list, resize_shape):
    num_of_img = len(input_list)
    H, W = mask.shape
    H_r, W_r = resize_shape
    # ratio when resized
    ratio = np.float(H*W)/(H_r*W_r)
    gt_density = np.zeros((num_of_img, H_r, W_r))
    gt_count = np.zeros(num_of_img)
    for i, loc_list in enumerate(input_list):
        # create temp org density
        gt_density_org = np.zeros((H, W))
        for loc in loc_list:
            # check if empty (no vehicle detected)
            if loc[0] == '-':
                continue
            else:
                if loc.shape[0] == 1:
                    temp_loc = loc[0]
                else:
                    temp_loc = loc
                    print(loc.shape)
                gt_density_org += generateSingle2DGaussian(mask.shape, temp_loc)
        # apply mask
        gt_density_org = np.multiply(gt_density_org, mask)
        # resize
        gt_density[i] = ratio * cv2.resize(gt_density_org, (224, 224))
        gt_count[i] = np.sum(gt_density[i])
        print(str(i+1) + ": " + str(np.sum(gt_density_org)))
    return gt_density, gt_count


def new_generateDensityMap(mask, input_list, resize_shape):
    num_of_img = len(input_list)
    H, W = mask.shape
    H_r, W_r = resize_shape
    # ratio when resized
    ratio = np.float(H*W)/(H_r*W_r)
    gt_density = np.zeros((num_of_img, H_r, W_r))
    gt_count = np.zeros(num_of_img)
    for i, loc_list in enumerate(input_list):
        # create temp org density
        gt_density_org = np.zeros((H, W))
        for loc in loc_list[0]:
            # check if empty (no vehicle detected)
            if loc[0] == '-':
                continue
            else:
                if loc.shape[0] == 1:
                    temp_loc = loc[0]
                else:
                    temp_loc = loc
                gt_density_org += generateSingle2DGaussian(mask.shape, temp_loc)
        # apply mask
        gt_density_org = np.multiply(gt_density_org, mask)
        # resize
        gt_density[i] = ratio * cv2.resize(gt_density_org, (224, 224))
        gt_count[i] = np.sum(gt_density[i])
        print(str(i+1) + ": " + str(np.sum(gt_density_org)))
    return gt_density, gt_count


def get_global_mean(img_path, img_type, img_num, num_of_channel, mask):
    img_path += '/*.'+img_type
    img_files = glob.glob(img_path)[0:img_num]
    roi = np.sum(mask)
    global_mean = np.zeros(num_of_channel)
    for i, img in enumerate(img_files):
        if i % 100 == 0:
            print(i)
        temp_img = cv2.imread(img)
        for j in range(num_of_channel):
            temp_img[:,:,j] = np.multiply(mask, temp_img[:,:,j])
            global_mean[j] += np.sum(temp_img[:,:,j])/roi
    return global_mean/img_num


#------------------------------------------------------------------
#----


# resize image (need to create the output path first) (not to 224*224, but to find the ROI)
'''
img_path = './2018-09-14 cmu_c2/*.jpg'
output_path = './2018-09-14 cmu_c2_img'
ratios = [0.8/6, 4.0/6, 1.0/5, 3.5/5]
img_shape = [1080, 1920]
resize_imgs_v2(img_path, output_path, ratios, img_shape)
'''


# image labelling
'''
index = '701-800'
img_path = './2018-09-14 cmu_c2_img/*.jpg'
label_list = img_labelling(img_path, index)
# convert to list object for numpy
label_list = np.asarray(label_list, dtype=list)
#print(label_list)
np.save('./2018-09-14 cmu_c2_label/label_'+index+'.npy', label_list)
'''



# ground truth process
'''
mask_dir = './mask/cmu_c2_cropped_roi.mat'
label_dir = './2018-09-14 cmu_c2_label/*.npy'
mask = loadmat(mask_dir).get('BW').astype(np.float)
# merge gt labels

label_files = glob.glob(label_dir)
print(label_files)

rang = 100
resize_shape = (224, 224)
rank = [int(s.split()[1].split('_')[-1].split('-')[0]) for s in label_files]
sorted_label_list = [label_files for _, label_files in sorted(zip(rank, label_files))]
print(sorted_label_list)
# merge labels
total_list = []
for i, f in enumerate(sorted_label_list):
    print(i)
    curr_arr = np.load(f)
    print(curr_arr.shape)
    for j in range(rang):
        total_list.append(curr_arr[j])

print(len(total_list))


print('get masks')
gt_density, gt_count = new_generateDensityMap(mask, total_list, resize_shape)
# save
save_dir = ''
np.save('./cmu_c2_data/2018-09-14_gt_density.npy', gt_density)
np.save('./cmu_c2_data/2018-09-14_gt_count.npy', gt_count)
'''




# get global mean
# consider mask
'''
mask_dir = './mask/cmu_c1_cropped_roi.mat'
img_path = './2018-07-05 cmu_c1_img'
img_type = 'jpg'
img_num = 1500
num_of_channel = 3
mask = loadmat(mask_dir).get('BW').astype(np.float)
global_mean = get_global_mean(img_path, img_type, img_num, num_of_channel, mask)
print(global_mean)

# global mean: [ 132.58914922  130.73264547  135.48747542], for 2018-06-01
# global mean: [ 144.44285073  147.65803605  156.58531561], for 2018-07-01
# global mean: [ 132.97366490  134.19162809  141.82866558], for 2018-07-02
# global mean: [ 143.90347464  147.07400714  155.36135867], for 2018-07-03
# global mean: [ 131.68339103  131.21488317  136.58659556], for 2018-07-05
'''



# prepare training image
'''
Objective:
    - for test img with global (unique) mask
    - convert RGB img to matrix (n, 224, 224, 3)
    - add mask
    - subtract global mean
    - reshape to 224*224
Input:
    - dim: (224, 224)
    - global_mean: np.array([132.58914922  130.73264547  135.48747542])
'''

'''
def img_2_array_test(img_path, img_type, mask, file_num, dim, global_mean):
    keyword = img_path +'/*.' + img_type
    f_list = glob.glob(keyword)
    data_array = np.zeros((file_num, dim[0], dim[1], 3))
    for i in range(file_num):
        img = cv2.imread(f_list[i])
        # subtract mean
        temp_array = np.array(img, dtype=np.float32) - global_mean
        # apply mask
        for j in range(3):
            temp_array[:,:,j] = np.multiply(temp_array[:,:,j], mask)
        # resize
        temp_array = cv2.resize(temp_array, dim)
        # RGB to BGR
        temp_array = temp_array[...,::-1]
        data_array[i,:,:,:] = temp_array
        if i % 100 == 0:
            print(i)
    return data_array


mask_dir = './mask/cmu_c2_cropped_roi.mat'
img_path = './2018-09-14 cmu_c2_img'
img_type = 'jpg'
mask = loadmat(mask_dir).get('BW').astype(np.float)
file_num = 800
dim_resize = (224, 224)
global_mean = np.array([0.0, 0.0, 0.0])


X_train = img_2_array_test(img_path, img_type, mask, file_num, dim_resize, global_mean)
np.save('./cmu_c2_data/2018-09-14-X_train.npy', X_train)
'''



'''
a = np.load('./cmu_c2_data/2018-09-06_gt_density.npy')
b = a[0]
b[np.where(b>0.0001)] = 100
cv2.imshow('test', b)
cv2.waitKey(0)
'''


