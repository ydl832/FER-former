# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:59:37 2021
@author: Yande
"""


class Config(object):
    def data_choose():

        # train_file = 'H:/Yande_Li/ExpressDataset/Facemer/train(FER+)_aligned.txt'
        # test_file = 'H:/Yande_Li/ExpressDataset/Facemer/test(FER+)_aligned.txt'
        train_file = 'H:/Yande_Li/ExpressDataset/Facemer/train(RAFDB).txt'
        test_file = 'H:/Yande_Li/ExpressDataset/Facemer/test(RAFDB).txt'
        # train_file = 'H:/Yande_Li/ExpressDataset/Facemer/train(SFEW).txt'
        # test_file = 'H:/Yande_Li/ExpressDataset/Facemer/test(SFEW).txt'
        return train_file, test_file

    exp_lr = 0.0001
    batch_size = 16
    epochs = 30
    momentum = 0.9
    step_size = 30  # adjust learning rate after 25 epoch
    gamma = 0.1     # lr =  lr/10
    outputdir = 'H:/Yande_Li/Facemer/saved_model'
    exp_dataset = 'RAFDB'
    device = 'cuda:0'
