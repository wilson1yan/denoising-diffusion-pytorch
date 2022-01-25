from ast import Mult
from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from tensorfn.data import LMDBReader
from torchvision import datasets


def get_dataset(conf, transform):
    if conf.dataset.name == 'cifar10':
        return CIFAR10(conf.dataset.path, train=True, transform=transform,
                       download=True)
    else:
        return MultiResolutionDataset(conf.dataset.path, transform, 
                                      conf.dataset.resolution)


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.reader = LMDBReader(path, reader="raw")

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        img_bytes = self.reader.get(
            f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
        )

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
