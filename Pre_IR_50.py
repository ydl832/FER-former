# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 04:48:00 2022

@author: Administrator
"""
import torch
import torchextractor as tx
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from f_vit import ViT
import numpy as np
from einops.layers.torch import Rearrange
from networks.model_irse import IR_50
import clip
from sklearn.manifold import TSNE
import random
device = "cuda" if torch.cuda.is_available() else "cpu"


def ir():
    model_ir = IR_50([112, 112])
    checkpoint = torch.load('./checkpoints/backbone_ir50_ms1m_epoch63.pth')
    model_ir.load_state_dict(checkpoint)
    return model_ir


def feature_clip():
    model_clip, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    text_encoder = model_clip.encode_text
    text_token = clip.tokenize
    return text_encoder, text_token


class ModelClassifier(nn.Module):
    def __init__(self):
        super(ModelClassifier, self).__init__()
        
        self.backbone = ir()
        self.text_encoder, self.text_token = feature_clip()
        self.text_encoder_t, self.text_token_t = feature_clip()
        self.token1 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
                                    nn.Linear((256 * 2 ** 2), 256)) #36
        self.token2 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=3, p2=3),
                                    nn.Linear((256 * 3 ** 2), 256)) #16
        self.token3 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
                                    nn.Linear((256 * 4 ** 2), 256)) #9
        self.token4 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=6, p2=6),
                                    nn.Linear((256 * 6 ** 2), 256)) #4
        self.token5 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=12, p2=12),
                                    nn.Linear((256 * 12 ** 2), 256)) #1
        self.vit = ViT(embed_dim=256, num_patches=50, depth=16, num_heads=4, mlp_ratio=4.0)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(512, 256)
        self.avg_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.model = tx.Extractor(self.backbone, ["body.20"])
                               
    def forward(self, x, label, phase):
        _, features = self.model(x)
        feature20 = features['body.20']    # 256*14*14
        feature20 = self.avg_pool(feature20)
        embed = torch.cat((self.token1(feature20), self.token3(feature20), self.token4(feature20), self.token5(feature20)), dim=1)
        # embed = self.token1(feature20)
        out, img_feature = self.vit(embed)

        text = "a face image of "
        text_t = self.text_token_t([text + "surprise", text + "fear",
                                    text + "disgust", text + "happy",
                                    text + "sad", text + "angry",
                                    text + "neutral"]).to(device)  # 7*77
        # text1 = " expression is shown on the image"
        # text_t = self.text_token_t(["a surprise" + text1, "a fear" + text1,
        #                             "a disgust" + text1, "a happy" + text1,
        #                             "a sad" + text1, "an angry" + text1,
        #                             "a neutral" + text1]).to(device) #7*77
        # text_feature = self.fc1(self.text_encoder(text).type(torch.FloatTensor).cuda())
        text_feature_t = self.fc2(self.text_encoder_t(text_t).type(torch.FloatTensor).cuda()) #7*256
        img_features = img_feature/img_feature.norm(dim=-1, keepdim=True)
        # text_features = text_feature/text_feature.norm(dim=-1, keepdim=True)
        text_features_t = text_feature_t/text_feature_t.norm(dim=-1, keepdim=True)
        values = (100.0 * img_features @ text_features_t.t())#.softmax(dim=-1)

        return out, img_features, _, values


def make_model():
    model = ModelClassifier()
    return model


