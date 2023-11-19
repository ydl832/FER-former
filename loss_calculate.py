# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 06:08:19 2022

@author: Administrator
"""

import torch.nn as nn
import torch
# import numpy as np
import torch.nn.functional as F


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.soft_max = nn.Softmax(dim=-1)

    def cross_entropy(self, predict_label, true_label):
        return torch.sum(- true_label * torch.log(predict_label))/len(true_label)
    
    def forward(self, sembed, logit, class_label, label, values):

        loss0 = self.criterion(sembed, label)
        loss1 = self.criterion(values, label)

        # label_gt = torch.arange(len(class_label)).cuda()
        # loss_i = (F.cross_entropy(logit, label_gt)) #*mask).sum()
        # loss_t = (F.cross_entropy(logit.T, label_gt)) #*mask).sum()
        # clip_loss = (loss_i + loss_t)/2

        return loss0 + loss1, loss1



