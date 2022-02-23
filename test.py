import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import batch_size, num_workers
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter

import torchvision
from torchvision.utils import save_image

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
            save_image(img, f'output/test/{index}img.jpg')
            with open(f'output/test/{index}out.txt', 'w') as f:
                for tensor in out:
                    out_list = str(tensor.tolist())
                    print(out_list)
                    f.write(out_list)

        # Calculate loss
        out = out.squeeze(dim=1)
        loss = criterion(out * 2, target)
        with open(f'output/test/{index}loss.txt', 'w') as f:
            f.write(str(loss.tolist()))

        losses.update(loss.item(), img.size(0))
        index = index + 1

    print('Elapsed: {0:.5f} ms'.format(elapsed / num_samples * 1000))
    print('Loss: {0:.2f}'.format(losses.avg))
    with open(f'output/test/avgloss.txt', 'w') as f:
            f.write(str(losses.avg))