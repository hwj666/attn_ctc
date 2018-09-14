import torch.utils.data as data
import torch.nn as nn
import cv2
import os
import numpy as np
import torch
from collections import OrderedDict
import pdb
class CrnnEncoder(nn.Module):
    def __init__(self,output_size,encode_H,n_layers=1,bidirectional=False):
        super(CrnnEncoder, self).__init__()
        self.cnn_feature = nn.Sequential(
            OrderedDict([
                # 32 x 100
                ('conv1', nn.Conv2d(1,64,3,1,1)),
                ('relu1', nn.ReLU(True)),
                ('pool1', nn.MaxPool2d(2,2)),
                # 16 x 50
                ('conv2', nn.Conv2d(64,128,3,1,1)),
                ('relu2', nn.ReLU(True)),
                ('pool2', nn.MaxPool2d(2,2)),
                # 8 x 25
                ('conv3_1', nn.Conv2d(128,256,3,1,1)),
                ('relu3_1', nn.ReLU(True)),
                ('conv3_2', nn.Conv2d(256,256,3,1,1)),
                ('relu3_2', nn.ReLU(True)),
                ('pool3', nn.MaxPool2d((2,1),2)),
                # 4 x 13
                ('conv4_1', nn.Conv2d(256,512,3,1,1)),
                ('BN4_1', nn.BatchNorm2d(512)),
                ('relu4_1', nn.ReLU(True)),
                ('conv4_2', nn.Conv2d(512,512,3,1,1)),
                ('BN4_2', nn.BatchNorm2d(512)),
                ('relu4_2', nn.ReLU(True)),
                ('pool4', nn.MaxPool2d((2,1),2)),
                # 2 x 7
                ('conv5', nn.Conv2d(512,512,2,1,0)),
                ('relu5', nn.ReLU(True)),
                # 1 x 6
            ])
        )
        self.encode = nn.GRU(512,encode_H,n_layers,bidirectional=bidirectional)
        if bidirectional:
            self.prob = nn.Linear(2*encode_H,output_size)
        else:
            self.prob = nn.Linear(encode_H,output_size)
        
    def forward(self,image):
        feature = self.cnn_feature(image)
        b,c,h,w = feature.size()
        assert h == 1, 'the height of conv must be 1'
        feature = feature.squeeze(2)
        feature = feature.permute(2,0,1) # [w,b,c] [seq_len,batch,n_feature]
        rnn_out,encoder_hidden = self.encode(feature)
        output = self.prob(rnn_out)

        return rnn_out,encoder_hidden,output

