import data_loader as dl
import transform as tsfm
from crnn import CrnnEncoder
from attention import AttnDecoder, attention_ctc
import torch
import editdistance
import os
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

def eval_model():

    for ratio in [0,2,5,8,10]:
        for start in [100,200,300,400,500,600,700,800,900,1000]: 
            restore_train = '../model_{}/model_{}.pth'.format(ratio,start)
            seq2seq.load_state_dict(torch.load(restore_train))
            for data_dir in ['ICDAR2003','IIIT5K_train','IIIT5K_test','svt_train','svt_test','']:
                if (start < 1000 and data_dir != ''):
                    continue
                data_path = os.path.join('../data/',data_dir)
                anno_path = os.path.join('../data/'+data_dir,'anno.txt')
                train_loader = dl.train_data(data_path,anno_path,batch_size=1)

                seq2seq.eval()
                # attn_edit,ctc_edit, ctc_len,attn_len = 0,0,0,0
                ctc_norm,attn_norm,sample_size = 0.0,0.0,0
                for sample in train_loader:
                    with torch.no_grad():
                        img = sample['image'].to(device)
                        ctc_res,attn_res, _ = seq2seq.predict(img)

                    label = ''.join(map(lambda w:tsfm.charset[w], sample['ctc_label'][0])).lower()
                    attn_distance = editdistance.eval(label,attn_res)
                    ctc_distance = editdistance.eval(label,ctc_res)

                    ctc_norm += ctc_distance*1.0 / len(label)
                    attn_norm += attn_distance*1.0 / len(label)
                    sample_size += 1
                    # attn_edit += attn_distance
                    # ctc_edit += ctc_distance
                    # ctc_len += len(ctc_res) + len(label)
                    # attn_len += len(attn_res) + len(label)
                # print 'model_{}_{}_{}_{}_{}'.format(ratio,start,data_dir,round(ctc_edit*1.0 / ctc_len,4),round(attn_edit*1.0 / attn_len,4))
                print 'model_{}_{}_{}_{}_{}'.format(ratio,start,data_dir,round(ctc_norm / sample_size,4),round(attn_norm / sample_size,4))

if __name__ == '__main__':
    eval_model()