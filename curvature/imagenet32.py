# Imagenet loader for torchvision <= 0.2.0

from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class IMAGENET32(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'Imagenet32_train_npz'
    url = "http://www.image-net.org/image/downsample/Imagenet32_train_npz.zip"
    filename = "Imagenet32_train_npz.zip"
    tgz_md5 = 'b0d308fb0016e41348a90f0ae772ee38'
    train_list = [
        ['train_data_batch_1.npz', '464fde20de6eb44c28cc1a8c11544bb1'],
        ['train_data_batch_2.npz', 'bdb56e71882c3fd91619d789d5dd7c79'],
        ['train_data_batch_3.npz', '83ff36d76ea26867491a281ea6e1d03b'],
        ['train_data_batch_4.npz', '98ff184fe109d5c2a0f6da63843880c7'],
        ['train_data_batch_5.npz', '462b8803e13c3e6de9498da7aaaae57c8'],
        ['train_data_batch_6.npz', 'e0b06665f890b029f1d8d0a0db26e119'],
        ['train_data_batch_7.npz', '9731f469aac1622477813c132c5a847a'],
        ['train_data_batch_8.npz', '60aed934b9d26b7ee83a1a83bdcfbe0f'],
        ['train_data_batch_9.npz', 'b96328e6affd718660c2561a6fe8c14c'],
        ['train_data_batch_10.npz', '1dc618d544c554220dd118f72975470c'],
    ]

    test_list = [
        ['val_data.npz', 'a8c04a389f2649841fb7a01720da9dd9'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = np.load(fo)
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels.extend(entry['labels'])
                else:
                    self.train_labels.extend(['fine_labels'])
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((1281167, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = np.load(fo)
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((50000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return 1281167
        else:
            return 50000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)