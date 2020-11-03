
import torch
from torchvision import datasets

import argparse
import os

######################################################################

parser = argparse.ArgumentParser(description='DLC prologue file for practical sessions.')

parser.add_argument('--full',
                    action='store_true', default=False,
                    help = 'Use the full set, can take ages (default False)')

parser.add_argument('--tiny',
                    action='store_true', default=False,
                    help = 'Use a very small set for quick checks (default False)')

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed (default 0, < 0 is no seeding)')

parser.add_argument('--cifar',
                    action='store_true', default=False,
                    help = 'Use the CIFAR data-set and not MNIST (default False)')

parser.add_argument('--data_dir',
                    type = str, default = None,
                    help = 'Where are the PyTorch data located (default $PYTORCH_DATA_DIR or \'./data\')')

# Timur's fix
parser.add_argument('-f', '--file',
                    help = 'quick hack for jupyter')

args = parser.parse_args()

if args.seed >= 0:
    torch.manual_seed(args.seed)

######################################################################
# The data

def load_data(train_size, test_size, cifar = None, normalize = False, flatten = True):

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = os.environ.get('PYTORCH_DATA_DIR')
        if data_dir is None:
            data_dir = './data'

    if args.cifar or (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)

        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)
        print(torch.from_numpy(cifar_train_set.data).float().shape)
        train_input = torch.from_numpy(cifar_train_set.data).float()

        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        print(train_input.shape)
        train_target = torch.tensor(cifar_train_set.targets, dtype = torch.int64)

        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype = torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_input = torch.functional.F.avg_pool2d(train_input, kernel_size = 2)
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_input = torch.functional.F.avg_pool2d(test_input, kernel_size = 2)
        test_target = mnist_test_set.targets

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if args.full:
        if args.tiny:
            raise ValueError('Cannot have both --full and --tiny')
    else:
        if args.tiny:
            print('** Reduce the data-set to the tiny setup')
            train_input = train_input.narrow(0, 0, 500)
            train_target = train_target.narrow(0, 0, 500)
            test_input = test_input.narrow(0, 0, 100)
            test_target = test_target.narrow(0, 0, 100)
        else:
            print('** Reduce the data-set (use --full for the full thing)')
            train_input = train_input.narrow(0, 0, train_size)
            train_target = train_target.narrow(0, 0, train_size)
            test_input = test_input.narrow(0, 0, test_size)
            test_target = test_target.narrow(0, 0, test_size)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))



    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target
