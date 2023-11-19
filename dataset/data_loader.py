# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:39:41 2021
@author: Yande
"""
from dataset.dataset_pair import ImageList
import torchvision.transforms as transforms
from config import Config
from torch.utils.data import DataLoader

# RAF-DB


def make_train_loader(filelist):
    normalize = transforms.Normalize(mean=[0.57514471, 0.44907006, 0.40058434],
                                     std=[0.20640279, 0.18882187, 0.18030375])
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing()])

    loader = DataLoader(
        ImageList(fileList=filelist, transform=transform),
        batch_size=Config.batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    return loader


def make_eval_loader(filelist):
    normalize = transforms.Normalize(mean=[0.57514471, 0.44907006, 0.40058434],
                                     std=[0.20640279, 0.18882187, 0.18030375])
    transform = transforms.Compose([          
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        normalize])

    loader = DataLoader(
        ImageList(fileList=filelist, transform=transform),
        batch_size=Config.batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    return loader
