import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import batch_size, num_workers, im_size
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter

import torchvision
from torchvision.utils import save_image
import numpy as np
import cv2 as cv
from numpy.linalg import inv

device = torch.device('cpu')

if __name__ == '__main__':
    filename = 'BEST_model.pt'

    print('loading {}...'.format(filename))
    model = MobileNetV2()
    model.load_state_dict(torch.load(filename))

    test_dataset = DeepHNDataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)

    num_samples = len(test_dataset)

    # Loss function
    criterion = nn.L1Loss().to(device)
    losses = AverageMeter()
    elapsed = 0

    # Batches
    index = 0
    for (img, target) in tqdm(test_loader):
        # Move to CPU, if available
        # img = F.interpolate(img, size=(img.size(2) // 2, img.size(3) // 2), mode='bicubic', align_corners=False)
        img = img.to(device)  # [N, 3, 128, 128]
        target = target.float().to(device)  # [N, 8]

        # Forward prop.
        with torch.no_grad():
            start = time.time()
            out = model(img)  # [N, 8]
            end = time.time()
            elapsed = elapsed + (end - start)
        
        img1, img2, _ = torch.unbind(img, dim=1)
        save_image(img, f'output/test/{index}in.jpg')
        save_image(img1, f'output/test/{index}in_og.jpg')
        save_image(img2, f'output/test/{index}in_warp.jpg')
        img1 = cv.imread(f'output/test/{index}in_og.jpg', 0)
        img2 = cv.imread(f'output/test/{index}in_warp.jpg', 0)
        out2 = out.squeeze(dim=1)

        size = (1920, 1920)
        size = (im_size, im_size)
        out_list = out2[0].tolist()
        x1 = np.array(out_list)
        x2 = np.array([256, 256, 256, 1280, 1280, 1280, 1280, 256])
        x2 = np.array([64, 64, 64, 320, 320, 320, 320, 64])
        seq = iter(np.add(x1, x2))
        four_points = [(256, 256), (256, 1280), (1280, 1280), (1280, 256)]
        four_points = [(64, 64), (64, 320), (320, 320), (320, 64)]
        perturbed_four_points = [(round(data), round(next(seq))) for data in seq]
        # [(256, 256), (256, 1280), (1280, 1280), (1280, 256)] <class 'list'> [(391, 244), (163, 1046), (1449, 1334), (1144, 473)] <class 'list'>
        #print(four_points, type(four_points), perturbed_four_points, type(perturbed_four_points))
        H = cv.getPerspectiveTransform(np.float32(perturbed_four_points), np.float32(four_points))
        H_inverse = inv(H)
        img3 = cv.warpPerspective(np.array(img1), H_inverse, size)
        cv.imwrite(f'output/test/{index}out.jpg', img3)
        
        # Calculate loss
        
        loss = criterion(out2 * 2, target)
        print(index, loss)
        # with open(f'output/test/{index}loss.txt', 'w') as f:
        #     f.write(str(loss.tolist()))

        with open(f'output/test/{index}out.txt', 'w') as f:
            for tensor in out:
                out_list = str(tensor.tolist())
                #print(out_list)
                f.write(f'Model Output: {out_list}\nTarget: {target[0].tolist()}\nOut2: {out2[0].tolist()}\nout2 * 2: {(out2 * 2)[0].tolist()}\nLoss: {loss}')

        losses.update(loss.item(), img.size(0))
        index = index + 1

    print('Elapsed: {0:.5f} ms'.format(elapsed / num_samples * 1000))
    print('Loss: {0:.2f}'.format(losses.avg))
    with open(f'output/test/avgloss.txt', 'w') as f:
            f.write(str(losses.avg))