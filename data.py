from pathlib import Path
import numpy as np
import rasterio
import math
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

from tools.utils import make_tuple


def get_pair_path(directory):
    label_orders = ['modis0','landsat0','modis1','landsat1']
    paths = []
    
    for label in label_orders:
        for path in Path(directory).glob('*.tif'):
            if path.name.startswith(label):
                paths.append(path)
                break
    return paths

def load_image_pair(directory):
   
    paths = get_pair_path(directory)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            images.append(im)
    return images

class PatchSet(Dataset):
   

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None):
        super(PatchSet, self).__init__()
       
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
       
        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]

        self.num_im_pairs = len(self.image_dirs)

        
        self.n_patch_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.n_patch_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y

    @staticmethod
    def transform(data):
        data[data < 0] = 0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        
        out=data.mul_(0.001)
        return out

    def map_index(self, index):
        
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n])
        patches = [None] * len(images)

        
        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0]),
                 id_y: (id_y + self.patch_size[1])]
            patches[i] = self.transform(im)

        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patch
