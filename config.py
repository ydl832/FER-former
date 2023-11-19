# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:59:37 2021
@author: Yande
"""


class Config(object):
    def data_choose():

        train_file = 'H:/Yande_Li/ExpressDataset/Facemer/train(RAFDB).txt'
        test_file = 'H:/Yande_Li/ExpressDataset/Facemer/test(RAFDB).txt'
        return train_file, test_file

    exp_lr = 0.0001
    batch_size = 16
    epochs = 30
    momentum = 0.9
    step_size = 30  
    gamma = 0.1     
    outputdir = 'H:/Yande_Li/Facemer/saved_model'
    exp_dataset = 'RAFDB'
    device = 'cuda:0'
