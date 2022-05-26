import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os

def make_dataset(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        lines = [l.strip() for l in lines]
        lines = [l.split(',') for l in lines]

    path = [l[0] for l in lines]
    valence = [float(l[1]) for l in lines]
    arousal = [float(l[2]) for l in lines]
    expr = [int(l[3]) for l in lines]
    AUs = [np.array([float(x) for x in l[4:]]) for l in lines]

    ids_list = [i for i, x in enumerate(AUs) if -1 not in x]
    AUs_new = [AUs[i] for i in ids_list]
    Val_new = [valence[i] for i in ids_list]
    Ars_new = [arousal[i] for i in ids_list]
    Exp_new = [expr[i] for i in ids_list]
    Pth_new = [path[i] for i in ids_list]

    data_list = [(Pth_new[i], [Val_new[i], Ars_new[i]], Exp_new[i], AUs_new[i]) for i in range(len(AUs_new))]
    return data_list

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class MTL_Dataset(Dataset):
    def __init__(self, root_path, train=True, transform=None, loader=default_loader):
        self._root_path = root_path
        self._train = train
        self._transform = transform
        self.loader = loader
        self.img_folder_path = os.path.join(root_path, 'cropped_aligned')
        if self._train:
            annotations_file = os.path.join(root_path, 'training_set_annotations.txt')
        else:
            annotations_file = os.path.join(root_path, 'validation_set_annotations.txt')
        
        self.data_list = make_dataset(annotations_file)

    def __getitem__(self, index):
        img, va, expr, au = self.data_list[index]
        img = self.loader(os.path.join(self.img_folder_path, img))

        if self._train:
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip)
        else:
            if self._transform is not None:
                img = self._transform(img)

        return img, va, expr, au

    def __len__(self):
        return len(self.data_list)
