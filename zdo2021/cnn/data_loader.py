import os
import pandas as pd
from torchvision.io import read_image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import pickle

class VarroaImageDataset(Dataset):
    def __init__(self, train, img_dir, transform=None, target_transform=None):
        self.train = train
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        with open(self.img_dir, 'rb') as f:
          self.images = pickle.load(f)

        if(self.train):
          self.img_labels = np.ones( [len(self.images)], dtype=int )
        else:
          self.img_labels = np.zeros( [len(self.images)], dtype=int )


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample
        
        
        
        
class VarroaPredictDataset(Dataset):
    def __init__(self, train, img, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.image = img

        if(self.train):
          self.img_labels = np.ones( [len(self.image)], dtype=int )
        else:
          self.img_labels = np.zeros( [len(self.image)], dtype=int )


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample
        
        