import os
import pickle
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm
from math import ceil
from numpy.linalg import inv
from config import train_file, valid_file, test_file, image_folder, im_size

debug_identity = False

def get_datum(img, test_image, size, rho, top_point, patch_size, f, index = 0, img2 = None):
    left_point = (top_point[0], patch_size + top_point[1])
    bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
    right_point = (patch_size + top_point[0], top_point[1])
    four_points = [top_point, left_point, bottom_point, right_point]
    
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
    
    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    
    try:
        H_inverse = inv(H)
    except:
        print(f'Not able to inv(H) {f}.\nH=\n{H}\nAttempting to continue without inverse.')
        H_inverse = H
    warped_image = cv.warpPerspective(img, H_inverse, size)
    warped_image = cv.resize(cv.imread(img2, 0), size)
    if debug_identity:
        warped_image = test_image
    
    # crop images
    #Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    #Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    # export
    cv.imwrite(f'output/preprocess/{index}img.jpg', test_image)
    cv.imwrite(f'output/preprocess/{index}perturb.jpg', warped_image)
    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    with open(f'output/preprocess/{index}out.txt', 'w') as f:
        f.write(f'four_points: {four_points}\nperturbed_four_points: {perturbed_four_points}\nH:\n{H}\nH_inverse:\n{H_inverse}\nH_four_points:\n{H_four_points}')

    training_image = np.dstack((test_image, warped_image))
    datum = (training_image, np.array(four_points), np.array(perturbed_four_points))
    return datum


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
    for f in tqdm(files):
        fullpath = os.path.join(image_folder, f)
        img = cv.imread(fullpath, 0)
        img = cv.resize(img, size)
        test_image = img.copy()
        if fullpath[-6:-4] == '29':
            continue
        img2 = f'data/images/transformImage{str(int(fullpath[-6:-4]) + 1).zfill(2)}.jpg'

        if not is_test:
            for top_point in [(0 + patch_size//8, 0 + patch_size//8), (patch_size//2 + patch_size//8, 0 + patch_size//8), (0 + patch_size//8, patch_size//2 + patch_size//8), (patch_size//2 + patch_size//8, patch_size*16//3 + patch_size//8), (patch_size//4 + patch_size//8, patch_size*8//3 + patch_size//8)]:
                datum = get_datum(img, test_image, size, rho, top_point, patch_size, f, index)
                samples.append(datum)
        else:
            top_point = (rho, rho)
            datum = get_datum(img, test_image, size, rho, top_point, patch_size, f, index, img2)
            samples.append(datum)
        index = index + 1

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not os.path.isdir('output/preprocess'):
        os.makedirs('output/preprocess')

    num_files = len(files)
    divisor = 1
    if len(files) > 70000:
        divisor = ceil(len(files)/70000)
    #np.random.shuffle(files)
    num_train_files = (num_files * 0.845)//divisor
    num_valid_files = (num_files * 0.07)//divisor
    num_test_files = (num_files * 0.085)//divisor
    num_train_files = num_valid_files = 0
    num_test_files = min(num_files, 50)

    if num_train_files + num_valid_files + num_test_files > num_files//divisor:
        print('The file split doesn\'t work. You might run out of RAM.')

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