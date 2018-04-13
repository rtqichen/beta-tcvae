import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        return x


class Dataset(object):
    def __init__(self, loc):
        self.dataset = torch.load(loc).float().div(255)

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        return self.dataset[index]


class Faces(Dataset):
    LOC = 'data/basel_face_renders.pth'

    def __init__(self):
        return super(Faces, self).__init__(self.LOC)

