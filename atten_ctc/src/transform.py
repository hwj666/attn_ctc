import torch.utils.data as data
from PIL import Image
import cv2
import os
import numpy as np
import torch
from config import CFG


charset_file = CFG['Charset_file']
charset = ['PAD','SOS','EOS']
with open(charset_file,'r') as lines:
    charset += map(lambda w: w.strip(), lines.readlines())

num_classes = len(charset)

class Fixscale(object):
    def __init__(self, output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
    
    def __call__(self, image,label):

        h,w = image.shape
        if isinstance(self.output_size,int):
            new_h = self.output_size
            new_w = new_h * w / h
        else:
            new_w,new_h = self.output_size
        image = cv2.resize(image,(new_w,new_h))

        return image,label

def CTC_Decode(encode):
    word = []
    for i in range(len(encode)):
        if encode[i] != 0 and (i == 0 or encode[i] != encode[i-1]):
            word.append(charset[encode[i]])
    return word


class Encode(object):
    def __init__(self, charset):
        self.charset = charset
        self.w_length = map(self.w_range, range(1024))

    def w_range(self, w):
        w1 = w / 2 # pool1
        w2 = w1 / 2 # pool2
        w3 = (w2 - 1) / 2 + 1 # pool3
        w4 = (w3 - 1) / 2 + 1 # pool4
        w5 = w4 - 2 + 1 # conv5
        return w5

    def __call__(self, image,label):

        label = map(lambda w: self.charset.index(w), label)
        label = np.asarray(label,np.int)

        h,w = image.shape
        min_w = self.w_length.index(len(label))

        if w < min_w:
            image = cv2.resize(image, (min_w,h))
        return image,label

class Normal(object):
    def __init__(self,mean_var):
        self.mean, self.var = mean_var
    
    def __call__(self, image,label):
        image = (image / 255.0 - self.mean) / self.var
        image = image.astype(np.float32)
        return image,label


class ToTensor(object):
    def __call__(self,image,label):
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image,label


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self,image,label):
        for t in self.transforms:
            image,label = t(image,label)
        return image,label

class ICDAR_tfsm(object):
    def __init__(self):
        self.FixHeight = CFG['FixHeight']
        self.mean_var = CFG['Mean_Var']
        self.composed = Compose([
            Fixscale(self.FixHeight),
            Encode(charset),
            Normal(self.mean_var),
            ToTensor()
        ])
    
    def __call__(self, image, label):
        return self.composed(image,label)