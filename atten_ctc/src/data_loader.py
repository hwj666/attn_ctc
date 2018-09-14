import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import torch
import transform as tsfm
import glob
from config import CFG

class train_dataset(object):
    def __init__(self,
                root_dir,
                samples_file,
                transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.isfile(samples_file):
            raise RuntimeError('samples file is not found')
        
        with open(samples_file,'r') as lines:
            self.samples = lines.readlines()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        sample = self.samples[index]
        img_path,label = sample.strip().split()
        image = cv2.imread(os.path.join(self.root_dir,img_path),0)
        
        if self.transform:
            image,label = self.transform(image,label)
        return image,label

class test_dataset(object):
    def __init__(self, root_dir):
        self.samples = glob.glob(root_dir + '*.jpg') + glob.glob(root_dir + '*.png')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        img_path = self.samples[index]
        image = cv2.imread(img_path,0)

        h,w = image.shape

        new_h = 32
        new_w = max(new_h * w / h,20)
        image = cv2.resize(image,(new_w,new_h))
        image = image[np.newaxis,...] / 255. - 0.5
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image, os.path.basename(img_path)

class PadCollate(object):

    def __init__(self,pattern):
        self.pattern = pattern

    def pad_image(self, img, max_w):
        img_h,img_w = img.size()
        pad_w = (max_w - img_w) / 2
        if self.pattern == 'resize':
            img = cv2.resize(img.numpy(), (max_w,img_h))
            img = torch.from_numpy(img)

        img = img.unsqueeze(0)
        pad_left = pad_w+1 if (max_w + img_w) % 2 else pad_w
        if self.pattern == 'replicate':
            img = F.pad(img,(pad_left,pad_w),'replicate')
        elif self.pattern == 'constant':
            img = F.pad(img,(pad_left,pad_w),'constant',0)

        return img

    def sos_pad_label(self, lab, max_length):
        lab_length = lab.size(0)
        pad_length = max_length - lab_length
        sos = torch.from_numpy(np.array([tsfm.charset.index('SOS')]))
        lab = torch.cat([sos.long(),lab.long(),torch.zeros(pad_length).long()])
        return lab

    def eos_pad_label(self, lab, max_length):
        lab_length = lab.size(0)
        pad_length = max_length - lab_length
        eos = torch.from_numpy(np.array([tsfm.charset.index('EOS')]))
        lab = torch.cat([lab.long(),eos.long(),torch.zeros(pad_length).long()])
        return lab

    def pad_collate(self,batch):
        new_image, new_label = zip(*batch)

        max_w = max(map(lambda x : x.size(1), new_image))
        new_image = torch.stack(map(lambda x:self.pad_image(x,max_w),new_image),dim=0)

        max_length = max(map(lambda x : x.size(0), new_label))

        input_label = torch.stack(map(lambda x: self.sos_pad_label(x,max_length),new_label),dim=0)
        input_label = input_label.transpose(0,1) # [T,B,C]

        target_label = torch.stack(map(lambda x: self.eos_pad_label(x,max_length),new_label),dim=0)
        target_label = target_label.transpose(0,1) # [T,B,C]

        features = {
            'image':new_image,
            'ctc_label':new_label,
            'input_label': input_label,
            'target_label':target_label,
            }
        return features

    def __call__(self,batch):
        return self.pad_collate(batch)

def train_data(data_path,anno_path,batch_size=128,method='resize',shuffle=True,num_workers=0):
    train = train_dataset(data_path,anno_path,tsfm.ICDAR_tfsm())
    train_loader = data.DataLoader(train,
                            batch_size=batch_size,
                            collate_fn=PadCollate(method),
                            shuffle=shuffle,num_workers=num_workers)
    return train_loader

def test_data(data_path):
    test = test_dataset(data_path)
    test_loader = data.DataLoader(test)
    return test_loader

if __name__ == '__main__':
    data = train_data()
    s = 0
    for sample in data:
        lab = sample['input_label']
        lab = lab.transpose(0,1)
        print lab
        