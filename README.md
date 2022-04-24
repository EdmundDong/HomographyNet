# HomographyNet

This is a deep convolutional neural network for estimating the relative homography between a pair of images. 
Deep Image Homography Estimation [paper](https://arxiv.org/abs/1606.03798) implementation in PyTorch.

## Features

- Backbone: MobileNetV2
- Dataset: MSCOCO 2014 training set and custom road images dataset

## Dependencies

- Python and PyTorch

Create new Conda environment:
```bash
$ conda env create --file environment.yml -n HomographyNet
```
Load dependencies into existing Conda environment:
```bash
$ conda env update --file environment.yml --prune
```

## Usage
### Data Pre-processing
Extract training images:
```bash
$ python3 extract.py
$ python3 pre_process.py
```

### Train
```bash
$ python3 train.py --lr 0.005 --batch-size 64
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

If you want to continue training after stopping, run in your terminal:
```bash
$ python3 train.py --lr 0.005 --batch-size 64 --checkpoint <checkpoint.tar>
```
examples: 
```bash
$ python3 train.py --lr 0.005 --batch-size 64 --checkpoint BEST_checkpoint.tar
$ python3 train.py --lr 0.005 --batch-size 64 --checkpoint checkpoint.tar --end-epoch 2000
$ python3 train.py --lr 0.005 --batch-size 64 --checkpoint checkpoint.tar --end-epoch 10000
```

## Test
Homography Estimation Comparison on Warped MS-COCO 14 Test Set.
```bash
$ python3 test.py
$ python3 test_orb.py --type surf
$ python3 test_orb.py --type identity
```
