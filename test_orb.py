import pickle

import cv2
import numpy as np
from tqdm import tqdm

MIN_MATCH_COUNT = 10


def compute_homo(img1, img2):
    try:
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # sift = cv2.xfeatures2d.SURF_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H
    except cv2.error as err:
        print(err)
    return np.identity(3)


def compute_mse(H, H_four_points):
    # print('H: ' + str(H))
    four_points = np.float32([[64, 64], [320, 64], [320, 320], [64, 320]])
    four_points = np.array([four_points])
    # print('four_points: ' + str(four_points))
    # print(four_points.shape)
    # print(H)
    pred_four_pints = cv2.perspectiveTransform(four_points, H)
    # print('predicted_four_pints: ' + str(pred_four_pints))
    # print(pred_four_pints.shape)
    # print('predicted_four_pints.shape: ' + str(predicted_four_pints.shape))
    error = np.subtract(np.array(pred_four_pints), np.array(four_points))
    # print('error: ' + str(error))
    # print('H_four_points: ' + str(H_four_points))
    mse = (np.square(error - H_four_points)).mean()
    return mse


def test():
    filename = 'data/test.pkl'
    with open(filename, 'rb') as file:
        samples = pickle.load(file)

    mse_list = []
    for sample in tqdm(samples):
        image, H_four_points = sample
        img1 = np.zeros((320, 320), np.uint8)
        img1[64:, 64:] = image[:, :, 0]
        img2 = np.zeros((320, 320), np.uint8)
        img2[64:, 64:] = image[:, :, 1]

        H = compute_homo(img1, img2)
        mse = compute_mse(H, H_four_points)
        mse_list.append(mse)

    print('MSE: {:5f}'.format(np.mean(mse_list)))
    print('len(mse_list): ' + str(len(mse_list)))


if __name__ == "__main__":
    test()
