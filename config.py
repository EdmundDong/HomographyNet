import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 600
batch_size = 1

#num_samples = 118287
#num_train = 500000
#num_valid = 41435
#num_test = 10000
image_folder = 'data/images'
train_file = 'data/train_images.pkl'
valid_file = 'data/valid_images.pkl'
test_file = 'data/test_images.pkl'

# Training parameters
num_workers = 8  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
