from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F

import pandas as pd
from PIL import Image as im


class UnlabeledDataset(Dataset):
    def __init__(self, df, dataset, transform=None, K=10):
        self.df = df
        self.dataset = dataset
        self.transform = transform
        self.K = K

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        index = self.dataset[idx]
        contents = self.df.loc[index]

        path = './../' + "/".join(contents['path'].split('\\'))
        img = im.open(path).convert('RGB')
        label = self.K

        if self.transform: img = self.transform(img)

        return img, label


class LabeledDataset(Dataset):
    def __init__(self, df, dataset, transform):
        self.df = df
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        index = self.dataset[idx]
        contents = self.df.loc[index]

        path = './../' + "/".join(contents['path'].split('\\'))
        img = im.open(path).convert('RGB')
        label = contents['label']

        if self.transform: img = self.transform(img)

        return img, label


class RepDataset(Dataset):
    def __init__(self, reps, labs, dataset):
        self.reps = reps
        self.labs = labs
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        index = self.dataset[idx]

        rep = self.reps[index]
        lab = self.labs[index]

        return rep, lab


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):

        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, label = self.dataset[item]
        y = 0
        if type(label) == np.ndarray: # If need to keep the label
            if label[1] == 1: # if label[1] == 1, this is a true label or high confidence prediction, Keep labels
                y = label[0]
            elif label[1] != 0: # for exp6
                if random.random() < self.p_uncond * (1-label[1]):
                    y = self.empty_token
                else:
                    y = label[0]
            elif random.random() < self.p_uncond: # set label none with probability p_uncond
                y = self.empty_token
            else: # keep the label if not set to none
                y = label[0]

        else: # if label is not a numpy array, then we don't need to keep labels
            if random.random() < self.p_uncond:
                y = self.empty_token
            else:
                y = label

        return x, np.int64(y)


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None
        self.unlabeled = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        elif split == "unlabeled":
            dataset = self.unlabeled
        else:
            raise ValueError

        return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# Cifars
class Cifars(DatasetFactory):
    def __init__(self, random_flip=False, cfg=False, p_uncond=None, u_data='id'):
        super().__init__()

        transform_train = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        transform_test = [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)

        if u_data == 'ood':
            df_name = f'Cifars_100labels_path'

        elif u_data == 'gen':
            df_name = 'Cifars_100labels_gen_WSSD_Pseudo_30depth_DPM'

        print("\n==> Load df from:", f'../data_path/{df_name}.csv')
        df = pd.read_csv(f'../data_path/{df_name}.csv')

        train_list = list(df[df['split'] == 'train'].index)
        test_list = list(df[df['split'] == 'test'].index)
        unlabeled_list = list(df[df['split'] == 'unlabeled'].index)

        self.K = max(list(df['label'].iloc[train_list])) + 1

        self.train = LabeledDataset(df, train_list, transform_train)
        self.unlabeled = UnlabeledDataset(df, unlabeled_list, transform_train, self.K)

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            # self.train = CFGDataset(self.train, p_uncond, self.K)
            # self.unlabeled = CFGDataset(self.unlabeled, p_uncond, self.K)

    @property
    def fid_stat(self):
        return 'assets/fid_stats/train_1000labels.npz'


def get_dataset(name, **kwargs):
    if name == 'Cifars':
        return Cifars(**kwargs)

    else:
        raise NotImplementedError(name)
