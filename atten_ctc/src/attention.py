import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import transform as tsfm

import pdb

class Attn(nn.Module):
    def __init__(self, method, encode_H, decode_H):
        super(Attn, self).__init__()
        self.method = method
        if self.method == 'dot':
            assert encode_H == decode_H, 'encode hidden size must be euql decode if use dot method'
        if self.method == 'general':
            self.attn = nn.Linear(encode_H, decode_H)
        elif self.method == 'concat':
            self.attn = nn.Linear((decode_H + encode_H), decode_H)
            self.v = nn.Parameter(torch.randn(encode_H))

    def forward(self, hidden, encoder_outputs):
        '''
        hidden:
            previous hidden state of the decoder in shape (1, B, H)
        encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        return
            attention energies in shape (B,T)
        '''
        
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,encode_H]
        attn_energies = self.score(hidden, encoder_outputs)
        attn_energies = F.softmax(attn_energies,dim=1)
        return attn_energies

    def score(self, hidden, encoder_outputs):
        '''
        hidden:
            [1, B, decode_H]
        encoder_outputs:
            [B, T, encode_H]
        '''
        if self.method == 'dot':
            energy = torch.bmm(hidden.transpose(0,1), encoder_outputs.transpose(1,2))
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)  # [B,T,decode_H]
            energy = torch.bmm(hidden.transpose(0,1), energy.transpose(1,2)) #[B,1,C]*[B,C,T]=[B,1,T]
        elif self.method == 'concat':
            T = encoder_outputs.size(1)
            hidden = hidden.repeat(T, 1, 1).transpose(0, 1)  # [B, T, decode_H]
            # [B,T,(encode_H+decode_h)] * [(encode_H+decode_h), decode_H] -> [B,T,decode_H]
            energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
            energy = energy.transpose(2, 1)  # [B,decode_H,T]
            
            v = self.v.repeat(encoder_outputs.data.shape[0], 1)  # [B,decode_H]
            v = v.unsqueeze(1)  # [B,1,decode_H]
            energy = torch.bmm(v, energy)  # [B,1,T]    
        else:
            raise RuntimeWarning('attention method is error')
        energy = F.softmax(energy.squeeze(1),dim=1)
        return energy  # [B,T]
class AttnDecoder(nn.Module):
    def __init__(self, method, encode_H, decode_H, output_size,n_layers=1,bidirectional=False):
        '''
        hidden_size : decode rnn hidden size
        output_size: word of num (one hot) (before word embedding size)
        context_size : encode output size (if not process ,it's encode hidden size)
        '''
        super(AttnDecoder, self).__init__()
        self.embeddimg = nn.Embedding(output_size, decode_H)
        if bidirectional:
            self.attn_model = Attn(method, encode_H*2, decode_H*2)
            self.out = nn.Linear((encode_H + decode_H)*2, output_size)
            self.gru = nn.GRU((decode_H+encode_H*2), decode_H, n_layers, bidirectional=bidirectional)

        else:
            self.attn_model = Attn(method, encode_H, decode_H)
            self.out = nn.Linear((encode_H + decode_H), output_size)
            self.gru = nn.GRU((decode_H+encode_H), decode_H, n_layers, bidirectional=bidirectional)

    def forward(self, word_input, last_hidden, last_context, encoder_outputs):
        '''
        every time step (T=1)
        word_input:
            input one hot word,will be embedded(self.embedding(word_input))
            shape [B] (embedd -> [B,hidden_size])
        last_hidden:
            as input to GRU 
            shape [?, B,hidden_size]
        last_context:
            shape [B, encode_H]
        encoder_outputs:
            all encode outputs, 
            [T,B,encode_H]
        '''
        word_embedded = self.embeddimg(word_input)  # [B, hidden_size]
        rnn_input = torch.cat([word_embedded, last_context], 1)
        rnn_input = rnn_input.unsqueeze(0) #[1,B,(hidden_size+encode_H)]
        
        # rnn_output [1,B,hidden_size]
        # hidden [num_layer*num_directions,B,hidden_size]
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        attn_weight = self.attn_model(rnn_output, encoder_outputs)  # [B,1,T]
        attn_weight = attn_weight.unsqueeze(1)  # [B,1,T]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,encode_H]
        context = torch.bmm(attn_weight, encoder_outputs)# [B,1,encode_H]
        context = context.squeeze(1)  # [B,encode_H]
        rnn_output = rnn_output.squeeze(0)  # [B,hidden_size]
        output = self.out(torch.cat([rnn_output, context],1))
        output = F.log_softmax(F.relu(output), dim=1) # [B,output_size]

        return output, hidden, context, attn_weight


class attention_ctc(nn.Module):
    def __init__(self, encoder, decoder):
        super(attention_ctc,self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, image, input_label, target_label):
        '''
        image 
            ocr input image shape[B,1,32,pad_W]
            feature shape [T,B,n_feature]->[w, B, 512]
        label
            ctc label
        input_label
            [T,B]
        '''
        outputs = []
        rnn_output, hidden, ctc_prob = self.encoder(image)
        batch_size = rnn_output.size(1)
        encode_H = rnn_output.size(2)
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        context = torch.zeros((batch_size,encode_H)).to(device)
        
        for output, tl in zip(input_label, target_label):
            output,hidden,context,_ = self.decoder(output,hidden,context,rnn_output)
            outputs.append(output.unsqueeze(0))
            teacher_force = random.random() < 0.5
            output = tl if teacher_force else output.max(1)[1]
            
        outputs = torch.cat(outputs)
        return ctc_prob, outputs
    
    def predict(self,image):
        rnn_output, hidden, ctc_prob = self.encoder(image)
        
        max_step = rnn_output.size(0)
        batch_size = rnn_output.size(1)
        encode_H = rnn_output.size(2)
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        rnn_input = torch.LongTensor([tsfm.charset.index('SOS')] * batch_size).to(device)
        context = torch.zeros((batch_size,encode_H)).to(device)
        
        outputs = []
        attn_weights = []
        for _ in range(max_step):
            output,hidden,context,attn_weight = self.decoder(rnn_input,hidden,context,rnn_output)
            rnn_input = output.max(1)[1]
            outputs.append(rnn_input)
            attn_weights.append(attn_weight)
        outputs = torch.cat(outputs)


        ctc_prob = ctc_prob.max(2)[1].transpose(0, 1)
        ctc_prob = ctc_prob.squeeze(0)
        ctc_res = tsfm.CTC_Decode(ctc_prob)
        ctc_res = ''.join(ctc_res).lower()


        attn_res = map(lambda w : tsfm.charset[w] if w > tsfm.charset.index('EOS') else '', outputs)
        attn_res = ''.join(attn_res).lower()

        return ctc_res, attn_res, attn_weights

    def beamsearch(self, image, beam_width=1):
        rnn_output, hidden, _ = self.encoder(image)
        
        max_step = rnn_output.size(0)
        batch_size = rnn_output.size(1)
        encode_H = rnn_output.size(2)
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')

        temp_input = [torch.LongTensor([tsfm.charset.index('SOS')] * batch_size).to(device)] * beam_width
        temp_context = [torch.zeros((batch_size,encode_H)).to(device)] * beam_width
        temp_hidden = [hidden] * beam_width
        temp_hidden = []

        for i in range(max_step):
            for j in range(beam_width):
                output,hidden,context,attn_weight = self.decoder(temp_input[j],temp_hidden[j],temp_context[j],rnn_output)

                rnn_input = output.sort(descending=True)[1][:beam_width]
                # for r_in in rnn_input:
        pass
                    


