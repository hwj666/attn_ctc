import data_loader as dl
import transform as tsfm
import numpy as np
import cv2
from crnn import CrnnEncoder
from attention import AttnDecoder, attention_ctc
from warpctc_pytorch import CTCLoss
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import pickle

ratio = 10
start = 0
restore_train = '../model_{}/model_{}.pth'.format(ratio, start)
restore_train = ''
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

encoder = CrnnEncoder(
    output_size=tsfm.num_classes,
    encode_H=256,
    n_layers=3,
    bidirectional=True).to(device)

decdoer = AttnDecoder(
    method='concat',
    encode_H=256,
    decode_H=256,
    output_size=tsfm.num_classes,
    n_layers=3,
    bidirectional=True).to(device)

seq2seq = attention_ctc(encoder, decdoer).to(device)

if restore_train != '':
    seq2seq.load_state_dict(torch.load(restore_train))


def attnctc_train(image, ctc_label, target_label, attn_outputs, ctc_pred, ctc_loss, attn_loss):
    pred_lens = torch.IntTensor([ctc_pred.size(0)] * ctc_pred.size(1))
    lab_lens = torch.IntTensor(map(lambda x: x.size(0), ctc_label))
    ctc_label = torch.cat(ctc_label).int()
    loss_ctc = ctc_loss(ctc_pred, ctc_label, pred_lens, lab_lens)
    attn_outputs = attn_outputs.contiguous()
    target_label = target_label.contiguous()
    loss_attn = attn_loss(
        attn_outputs.view(-1, attn_outputs.size(2)), target_label.view(-1))
    r = ratio * 0.1
    return (1-r)*loss_attn+r*loss_ctc.to(device)


def train_model():
    data_path = '../data/'
    anno_path = '../data/all_anno.txt'
    train_loader = dl.train_data(data_path,anno_path,num_workers=2)
    ctc_loss = CTCLoss().to(device)
    attn_loss = nn.NLLLoss(reduction='sum').to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=1e-4)
    sum_loss = 0
    sample_size = 0
    loss_list = []
    for epoch in range(1,1001):
        seq2seq.train()
        for sample in train_loader:
            img = sample['image'].to(device)
            target_label = sample['target_label'].to(device)
            input_label = sample['input_label'].to(device)
            ctc_label = sample['ctc_label']

            ctc_prob, attn_outputs = seq2seq(img, input_label, target_label)
            
            loss = attnctc_train(
                img, ctc_label, target_label, attn_outputs, ctc_prob, ctc_loss, attn_loss)
            sum_loss += loss.item()
            sample_size += img.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = sum_loss / sample_size
        print('epoch : {} loss : {}'.format(epoch, mean_loss))
        loss_list.append(mean_loss)
        if epoch % 100 == 0:
            torch.save(seq2seq.state_dict(),
                       '../model_{}/model_{}.pth'.format(ratio, epoch+start))
    print loss_list
    with open('loss.txt','wb') as f:
        for i in loss_list:
            f.write(str(i) + '\n')

def attn_test():
    test_loader = dl.test_data('../test/')
    seq2seq.eval()
    for img, name in test_loader:
        res = []
        with torch.no_grad():
            img = img.to(device)
            outputs, attn_weights = seq2seq.predict(img)
            # print attn_weights
            for i in outputs:
                if i >= tsfm.charset.index('EOS'):
                    res.append(tsfm.charset[i])
                else:
                    break
        print res,name

def test_model():
    test_loader = dl.test_data('../test/')
    decode = tsfm.CTC_Decode(tsfm.charset)
    encoder.eval()
    for img, name in test_loader:
        with torch.no_grad():
            img = img.to(device)
            _,__,preds = encoder(img)  # length x batch x num_letters
            preds = preds.max(2)[1].transpose(0, 1)  # batch x length
            preds = preds.squeeze(0)  # length
            preds = decode(preds)
            print(preds, name)


if __name__ == '__main__':
    train_model()