import os
import pickle
import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from config import image_folder
from config import train_file, valid_file, test_file

from math import floor


def get_datum(img, test_image, size, rho, top_point, patch_size, f, index = 0):
    left_point = (top_point[0], patch_size + top_point[1])
    bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
    right_point = (patch_size + top_point[0], top_point[1])
    four_points = [top_point, left_point, bottom_point, right_point]
    # print('top_point: ' + str(top_point))
    # print('left_point: ' + str(left_point))
    # print('bottom_point: ' + str(bottom_point))
    # print('right_point: ' + str(right_point))
    # print('four_points: ' + str(four_points))

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    # debug images
    try:
        H_inverse = inv(H)
    except:
        print(f'Not able to inv(H) {f}.\nH=\n{H}\nAttempting to continue without inverse.')
        H_inverse = H
        

    warped_image = cv.warpPerspective(img, H_inverse, size)

    # print('test_image.shape: ' + str(test_image.shape))
    # print('warped_image.shape: ' + str(warped_image.shape))

    Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    training_image = np.dstack((Ip1, Ip2))
    cv.imwrite(f'output/preprocess/{index}img.jpg', img)
    cv.imwrite(f'output/preprocess/{index}warp.jpg', warped_image)
    cv.imwrite(f'output/preprocess/{index}cimg.jpg', Ip1)
    cv.imwrite(f'output/preprocess/{index}cwarp.jpg', Ip2)
    # H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    datum = (training_image, np.array(four_points), np.array(perturbed_four_points))
    return datum


### This function is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb
def process(files, is_test):
    if is_test:
        size = (1920, 1088)
        # Data gen parameters
        patch_size = 1088
        rho = int(patch_size/4)

    else:
        size = (320, 240)
        # Data gen parameters
        rho = 32
        patch_size = 128

    samples = []
    index = 0
    for f in tqdm(files):
        fullpath = os.path.join(image_folder, f)
        img = cv.imread(fullpath, 0)
        #img = cv.resize(img, size)
        test_image = img.copy()

        if not is_test:
            for top_point in [(0 + 32, 0 + 32), (128 + 32, 0 + 32), (0 + 32, 48 + 32), (128 + 32, 48 + 32),
                              (64 + 32, 24 + 32)]:
                # top_point = (rho, rho)
                datum = get_datum(img, test_image, size, rho, top_point, patch_size, f, index)
                samples.append(datum)
        else:
            top_point = (rho, rho)
            datum = get_datum(img, test_image, size, rho, top_point, patch_size, f, index)
            samples.append(datum)
        index = index + 1

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    #divisor = 7
    divisor = 1
    n_files = len(files)//divisor

    files = files[:n_files] # only work part of training set. not enough memory to train it all in one go
    np.random.shuffle(files)

    num_files = len(files)
    print('num_files: ' + str(num_files) + str(files))

    num_train_files = floor(int(num_files * 0.845)//divisor)
    num_valid_files = floor(int(num_files * 0.07)//divisor)
    num_test_files = floor(int(num_files * 0.085)//divisor)
    num_train_files = num_valid_files = 0
    num_test_files = num_files

    if num_train_files + num_valid_files + num_test_files > n_files:
        print('The file split doesn\'t work.')

    train_files = files[:num_train_files]
    valid_files = files[num_train_files:num_train_files + num_valid_files]
    test_files = files[num_train_files + num_valid_files:num_train_files + num_valid_files + num_test_files]

    train = process(train_files, False)
    valid = process(valid_files, False)
    test = process(test_files, True)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))
    print('num_test: ' + str(len(test)))

    with open(train_file, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_file, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test, f)
