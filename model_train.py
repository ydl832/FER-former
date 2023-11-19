# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:29:26 2021
@author: Yande
"""
import torch
from torch.autograd import Variable
import os
from config import Config
# from Plot_confusion_matrix import plot_matrix
device = torch.device("cuda:0")


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    best_epoch = 0
    best_acc = 0.0
    model = model.to(device)
    criterion = criterion.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['test', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = test_loader
            running_loss = 0.0
            clip_losses = 0.0
            correct_vit = 0
            correct_clip = 0
            total = 0
            for n_iter, (inputs, text_label, label) in enumerate(data_loader):
                inputs, label = inputs.to(device), label.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    sembed, img_features, text_features, values = model(inputs, text_label, phase)

                    _, preds_vit = torch.max(sembed, 1)
                    _, preds_clip = torch.max(values, 1)
                    loss, clip_loss = criterion(sembed, _, _, label, values)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计
                running_loss += loss.item()
                clip_losses += clip_loss.item()
                correct_vit += (preds_vit == label).sum().float()
                correct_clip += (preds_clip == label).sum().float()
                total += len(label)

            epoch_loss = running_loss / (n_iter + 1)
            epoch_loss_c = clip_losses / (n_iter + 1)
            epoch_acc_vit = correct_vit.data / total
            epoch_acc_clip = correct_clip.data / total

            if phase == 'test' and (epoch_acc_clip > best_acc or epoch_acc_vit > best_acc):
                if epoch_acc_clip > epoch_acc_vit:
                    best_acc = epoch_acc_clip
                    # plot_matrix(all_targets, all_predicted_clip, best_acc)
                else:
                    best_acc = epoch_acc_vit
                    # plot_matrix(all_targets, all_predicted_vit, best_acc)
                torch.save(model.state_dict(), os.path.join(Config.outputdir, Config.exp_dataset+'.pth'))
            LR = optimizer.state_dict()['param_groups'][0]['lr']
            print('{} Loss: {:.4f}, clip_loss: {:.4f}, Acc_vit: {:.4f}, Acc_clip: {:.4f}, Lr: {:.2e}'.format(
                  phase, epoch_loss, epoch_loss_c, epoch_acc_vit, epoch_acc_clip, LR))
            if phase == 'test':
                print('Best Acc_clip: {:.4f}'.format(best_acc))
        print('-' * 50)

    file = open('C:/Users/Xin/Desktop/Best_acc.txt', 'a')
    file.write(str(best_acc) + ' ' + str(best_epoch))
    file.write('\n')
    file.close()

    return best_acc




