import cv2 as cv
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

def transform_func(data):
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop((256, 256), scale=(0.08, 1.0)),
        transforms.ToTensor(),
    ])
    return data_transform(data)

def generate_filelist():
    file_list = []
    root_dir = '../data/tiny-imagenet-200'
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg')):
                file_path = os.path.join(root, file)
                file_path = "/".join(file_path.split('/')[3:])
                file_list.append(file_path)
    with open('filelist.txt', 'w') as file:
        for path in file_list:
            file.write(path + '\n')

#generate_filelist()

class TinyImageNet(Dataset):
    def __init__(self, root_dir, filelist, train = True, transform = False) -> None:
        super().__init__()
        self.root = root_dir  # data directory
        self.filelist = self.load_files(filelist)
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.filelist[index])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = transform_func(image)
        else: 
            trf = transforms.Compose([
                transforms.Resize(size=[256, 256]),
                transforms.ToTensor(),
            ])
            image = trf(image)
        return image
    
    def load_files(self, filelist):
        # load file list
        #print(filelist)
        with open(filelist, 'r') as file:
            file_list = file.read().splitlines()
        return file_list

    def __len__(self):
        return len(self.filelist)