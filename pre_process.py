import os
import pickle
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
from math import ceil
from numpy.linalg import inv
from natsort import natsorted
from PIL import Image, ImageOps
from config import train_file, valid_file, test_file, image_folder, im_size
from torch import from_numpy
from torchvision.utils import save_image

debug_identity = False
generate_series = False

### This function is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb
def process(files, is_test):
    
    # Data gen parameters
    size = (im_size, im_size)
    if is_test:
        #size = (640, 480)
        #patch_size = 256
        patch_size = im_size
        rho = patch_size//4
    else:
        #size = (320, 240)
        #patch_size = 128
        size = (im_size//2, im_size//2)
        patch_size = im_size
        rho = patch_size//4

    samples = []
    index = 0
    for (f1, f2, m) in tqdm(files):
        fullpath1 = os.path.join(image_folder, f1)
        fullpath2 = os.path.join(image_folder, f2)
        #img1 = cv.imread(fullpath1, 0)
        #img2 = cv.imread(fullpath2, 0)
        #img1 = cv.resize(img1, size)
        #img2 = cv.resize(img2, size)
        img1 = Image.open(fullpath1)
        img2 = Image.open(fullpath2)
        img1 = ImageOps.grayscale(img1)
        img2 = ImageOps.grayscale(img2)
        color = 'black'
        img1 = ImageOps.pad(img1, size, color=color)
        img2 = ImageOps.pad(img2, size, color=color)
        #img1.show()
        #img2.show()
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)

        if generate_series:
            num_transforms = 20
            for index in range(num_transforms + 1):
                offset = size[0] * 2 * index/num_transforms
                shift_top = offset
                shift_bottom = offset/8
                drop_top = offset
                drop_bottom = offset/16
                four_points = [(0,0),(0,size[0]),(size[0],size[0]),(size[0],0)]
                perturbed_four_points = [(-shift_top,-drop_top),(shift_bottom,size[0]-drop_bottom),(size[0]-shift_bottom,size[0]-drop_bottom),(size[0]+shift_top,-drop_top)]
                H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
                warped_image = cv.warpPerspective(img1, inv(H), size)
                #cv.imshow('img1', img1)
                #cv.imshow('warped_image', warped_image)
                #cv.waitKey()
                cv.imwrite(f'output/preprocess/{index}img.jpg', img1)
                cv.imwrite(f'output/preprocess/{index}perturb.jpg', warped_image)
                with open(f'output/preprocess/{index}out.txt', 'w') as f:
                    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
                    target = np.reshape(H_four_points, (8,))
                    f.write(f'target: {target}\nH:\n{H}\ninv(H):\n{inv(H)}')
                training_image = np.dstack((img1, warped_image))
                samples.append((training_image, np.array(four_points), np.array(perturbed_four_points)))
        else:
            if debug_identity:
                img2 = img1
            training_image = np.dstack((img1, img2))
            
            four_points = np.zeros((4,2))
            perturbed_four_points = np.zeros((4,2))
            with open(os.path.join(image_folder, m), 'r') as file:
                for point in range(4):
                    four_points[point] = file.readline().strip().split(' ')
                for point in range(4):
                    perturbed_four_points[point] = file.readline().strip().split(' ')
            # print(m)
            # print(four_points)
            # print(perturbed_four_points)

            training_image0 = training_image[:, :, 0]
            training_image0 = cv.resize(training_image0, (im_size, im_size))
            training_image1 = training_image[:, :, 1]
            training_image1 = cv.resize(training_image1, (im_size, im_size))
            training_image2 = np.zeros((im_size, im_size, 3), np.float32)
            training_image2[:, :, 0] = training_image0 / 255.
            training_image2[:, :, 1] = training_image1 / 255.
            training_image2 = np.transpose(training_image2, (2, 0, 1))  # HxWxC array to CxHxW
            save_image(from_numpy(training_image2), f'output/preprocess/{index}img.jpg')
            with open(f'output/preprocess/{index}matrix.txt', 'w') as f:
                H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
                target = np.reshape(H_four_points, (8,))
                f.write(f'target: {target}\nfour_points:\n{four_points}\nperturbed_four_points:\n{perturbed_four_points}')
            samples.append((training_image, np.array(four_points), np.array(perturbed_four_points)))
        
        index = index + 1

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.csv'))]
    files = natsorted(files) # sort files in natural order
    if not os.path.isdir('output/preprocess'):
        os.makedirs('output/preprocess')

    # create tuples of images and matricies that go together, new tuples are `files`
    # tuple form: (f1, m, f2) for file1, matrix, file2
    #print(f'files{files}')
    files_new = []
    if generate_series:
        for index in range(0, len(files)):
            files_new.append((files[index], None, files[index]))
    else: 
        for index in range(1, len(files), 3):
            #print(f'{files[index - 1]}, {files[index]}, {files[index + 1]}')
            files_new.append((files[index - 1], files[index], files[index + 1]))
    files = files_new

    num_files = len(files)
    divisor = 1
    if len(files) > 70000:
        divisor = ceil(len(files)/70000)
    np.random.shuffle(files)
    ratio_train = ratio_valid = ratio_test = 0
    ratio_train = 0.845
    ratio_valid = 0.07
    ratio_test = 0.085
    ratio_ratio = 1 / (ratio_train + ratio_valid + ratio_test)
    num_train_files = int(round((num_files * ratio_train * ratio_ratio)/divisor))
    num_valid_files = int(round((num_files * ratio_valid * ratio_ratio)/divisor))
    num_test_files = int((num_files * ratio_test * ratio_ratio)/divisor)
    #num_train_files = num_valid_files = 0
    #num_test_files = min(num_files, 50)

    #print(f'num_files={num_files}, num_train_files={num_train_files}, num_valid_files={num_valid_files}, num_test_files={num_test_files}')

    if num_train_files + num_valid_files + num_test_files > num_files//divisor:
        print('File split invalid.')
        exit(1)

    train_files = files[:num_train_files]
    valid_files = files[num_train_files:num_train_files + num_valid_files]
    test_files = files[num_train_files + num_valid_files:num_train_files + num_valid_files + num_test_files]

    # print(f'train_files{train_files}')
    # print(f'valid_files{valid_files}')
    # print(f'test_files{test_files}')

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
    # with open(test_file, 'wb') as f:
    #     pickle.dump(test, f)