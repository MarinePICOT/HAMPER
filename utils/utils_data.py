from __future__ import division, absolute_import, print_function
from art.utils import load_cifar10
from subprocess import call

import os
import numpy as np
import scipy.io as sio
import torch


def from_dataloader_to_numpy(dataloader, index=0):
    cache_list = list(iter(dataloader))

    assert len(cache_list) > 0
    assert index < len(cache_list[0])

    result_list = np.array(list(map(lambda x: x[index].numpy(), cache_list)))

    return result_list

def from_numpy_to_dataloader(X, y, batch_size=100, shuffle=False):
    from torch.utils.data import DataLoader, TensorDataset

    tensor_x = torch.Tensor(X.astype(float)) if isinstance(X, np.ndarray) else X  # transform to torch tensor
    tensor_y = torch.Tensor(y.astype(float)) if isinstance(y, np.ndarray) else y

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # create your dataloader

    return dataloader


def get_attack_name_to_load(attack, epsilon):
    import re

    if re.match(r'.*(fgsm|pgd|bim)', attack):
        assert epsilon is not None, "'epsilon' cannot be None"
        attack_ = 'CE_' + attack + '_' + str(epsilon)
    else:
        attack_ = '_' + attack

    return attack_

def load_adv_data(dataset_name, attack, epsilon=None, adaptive=False, **kwargs):
    from setup import adv_data_dir

    attack_ = get_attack_name_to_load(attack, epsilon)

    if adaptive:
        adv_file_path = f"{adv_data_dir}/{dataset_name}/white-box-hamper/{dataset_name}{attack_}.npy"
    else:
        adv_file_path = f"{adv_data_dir}/{dataset_name}/{dataset_name}{attack_}.npy"

    return np.load(adv_file_path), attack_

def load_data(dataset_name):
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        num_classes = 10

    elif dataset_name == 'svhn':
        def load_svhn_data():
            os.makedirs("data/svhn/", exist_ok=True)
            if not os.path.isfile("data/svhn/train_32x32.mat"):
                print('Downloading SVHN train set...')
                call(
                    "curl -o data/svhn/train_32x32.mat "
                    "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                    shell=True
                )
            if not os.path.isfile("data/svhn/test_32x32.mat"):
                print('Downloading SVHN test set...')
                call(
                    "curl -o data/svhn/test_32x32.mat "
                    "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                    shell=True
                )
            train = sio.loadmat('data/svhn/train_32x32.mat')
            test = sio.loadmat('data/svhn/test_32x32.mat')
            x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
            x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
            y_train = np.reshape(train['y'], (-1,))
            y_test = np.reshape(test['y'], (-1,))
            np.place(y_train, y_train == 10, 0)
            np.place(y_test, y_test == 10, 0)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train = np.reshape(x_train, (73257, 32, 32, 3))
            x_test = np.reshape(x_test, (26032, 32, 32, 3))

            min = x_test.min()
            max = x_test.max()
            return (x_train, y_train), (x_test, y_test), min, max

        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_svhn_data()
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32) / 255.
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.
        num_classes = 10
        y_train, y_test = toCat_onehot(y_train, y_test, num_classes)

    elif dataset_name == 'cifar100':
        def load_cifar100_data(batch_size=500, shuffle=False):
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            data_man = datasets.CIFAR100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = data_man('data/', train=True, download=True, transform=transform_train)
            train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            x_train = from_dataloader_to_numpy(train_set)
            x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]))
            y_train = np.asarray(train_dataset.targets)

            test_dataset = data_man('data/', train=False, download=True, transform=transform_test)
            test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
            x_test = from_dataloader_to_numpy(test_set)
            x_test = x_test.reshape((x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3], x_test.shape[4]))
            y_test = np.asarray(test_dataset.targets)

            min = x_test.min()
            max = x_test.max()
            return (x_train, y_train), (x_test, y_test), min, max

        (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar100_data()
        num_classes = 100
        y_train, y_test = toCat_onehot(y_train, y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes


def toCat_onehot(y_train, y_test, numclasses):
    y_test_one_hot = np.zeros((len(y_test), numclasses))
    for i in range(len(y_test)):
        y_test_one_hot[i, y_test[i]] = 1

    y_train_one_hot = np.zeros((len(y_train), numclasses))
    for i in range(len(y_train)):
        y_train_one_hot[i, y_train[i]] = 1

    return y_train_one_hot, y_test_one_hot
