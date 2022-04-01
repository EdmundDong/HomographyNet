import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

import os
import time
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from torch import tensor
from numpy.linalg import inv
from utils import AverageMeter
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from config import batch_size, num_workers, im_size, test_file, device, image_folder, output_folder
all_H = []
all_H_files = []
stacks = []

def stack_image(img1, img2):
    stacks.append((img1, img2))
    img1 = cv.resize(cv.imread(img1, 0), (im_size, im_size))
    img2 = cv.resize(cv.imread(img2, 0), (im_size, im_size))
    img = np.dstack((img1, img2))
    #img = tensor(img)
    return img

def data_export(index, img1_loc, img2_loc):
    size = (im_size, im_size)
    out_list = out2[0].tolist()
    x1 = np.array(out_list)
    x2 = np.array([256, 256, 256, 1280, 1280, 1280, 1280, 256])
    x2 = np.array([64, 64, 64, 320, 320, 320, 320, 64])
    seq = iter(np.add(x1, x2))
    four_points = [(256, 256), (256, 1280), (1280, 1280), (1280, 256)]
    four_points = [(64, 64), (64, 320), (320, 320), (320, 64)]
    perturbed_four_points = [(round(data), round(next(seq))) for data in seq]
    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)
    img3 = cv.warpPerspective(np.array(img1).copy(), H_inverse, size)
    cv.imwrite(f'output/test_{output_folder}/{index}out.jpg', img3)
    img4 = cv.imread(f'output/test_{output_folder}/{index}in_og.jpg', 0)
    img5 = cv.imread(f'output/test_{output_folder}/{index}out.jpg', 0)
    img6 = np.zeros((im_size, im_size, 3), np.float32)
    img6[:, :, 0] = img4 / 255.
    img6[:, :, 1] = img5 / 255.
    img6 = torch.from_numpy(np.array([np.transpose(img6, (2, 0, 1))]))
    save_image(img6, f'output/test_{output_folder}/{index}out2.jpg')
    with open(f'output/test_{output_folder}/{index}out.txt', 'w') as f:
        for tensor in out:
            f.write(f'Model Output: {str(tensor.tolist())}\nTarget: {target[0].tolist()}\nOut2: {out2[0].tolist()}\nout2 * 2: {(out2*2)[0].tolist()}\nLoss: {loss}\n')
            f.write(f'H:\n{H}\nH:\n{np.array2string(H, suppress_small=True)}\nH_inverse:\n{H_inverse}\nH_inverse:\n{np.array2string(H_inverse, suppress_small=True)}\n')
            f.write(f'H_four_points:\n{np.subtract(np.array(perturbed_four_points), np.array(four_points))}')
    all_H.append(f'{index}\n{H}\n')
    all_H_files.append(f'{index} ({img1_loc} and {img2_loc})\n{H}\n')

if __name__ == '__main__':
    begin = time.time()
    filename = 'BEST_model.pt'

    print('loading {}...'.format(filename))
    model = MobileNetV2()
    model.to(device)
    model.load_state_dict(torch.load(filename))

    images = [f'{image_folder}/{file}' for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    samples = []
    print(images)
    for index in range(len(images) - 1):
        samples.append((stack_image(images[index], images[index + 1]), tensor(np.zeros(8)), tensor(np.zeros(8))))
    
    with open(test_file, 'wb') as f:
        pickle.dump(samples, f)

    test_dataset = DeepHNDataset(test_file)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_samples = len(test_dataset)

    # Loss function
    criterion = nn.L1Loss().to(device)
    losses = AverageMeter()
    elapsed = 0

    if not os.path.isdir(f'output/test_{output_folder}'):
        os.makedirs(f'output/test_{output_folder}')

    # Batches
    index = 0
    for (img, target) in tqdm(test_loader):
        img = img.to(device)  # [N, 3, 128, 128]
        target = target.float().to(device)  # [N, 8]

        # Forward prop.
        with torch.no_grad():
            start = time.time()
            out = model(img)  # [N, 8]
            elapsed = elapsed + (time.time() - start)
        
        img1, img2, _ = torch.unbind(img, dim=1)
        img1_loc, img2_loc = stacks[index]
        save_image(img, f'output/test_{output_folder}/{index}in.jpg')
        save_image(img1, f'output/test_{output_folder}/{index}in_og.jpg')
        save_image(img2, f'output/test_{output_folder}/{index}in_warp.jpg')
        img1 = cv.imread(f'output/test_{output_folder}/{index}in_og.jpg', 0)
        img2 = cv.imread(f'output/test_{output_folder}/{index}in_warp.jpg', 0)
        
        # Calculate loss
        out2 = out.squeeze(dim=1)
        loss = criterion(out2*2, target)
        losses.update(loss.item(), img.size(0))

        data_export(index, img1_loc, img2_loc)
        #print(f'Loss for {index}: {loss}')
        index = index + 1

    print('Avg Inference Time: {0:.5f} ms'.format(elapsed / num_samples * 1000))
    print('Loss: {0:.2f}'.format(losses.avg))
    print(f'Elapsed Time: {round(time.time() - begin, 4)} s')
    with open(f'output/test_{output_folder}/avgloss.txt', 'w') as f:
        f.write(str(losses.avg))
    with open(f'output/test_{output_folder}/all_H.csv', 'w') as f:
        f.write(''.join(all_H))
    with open(f'output/test_{output_folder}/all_H_files.csv', 'w') as f:
        f.write(''.join(all_H_files))