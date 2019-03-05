
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Encoder(nn.Module):
    def __init__(self,embed_size,hidden_size,n_layers=1,dropout=0.5):
        super(Encoder,self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(embed_size,hidden_size,n_layers,dropout=dropout,bidirectional=False)

    def forward(self, x_in,hidden=None):
        # x_in: (seq_len,batch,embed_size)

        outputs,last_hidden = self.gru(x_in,hidden)

        return outputs,last_hidden

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention,self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2,hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv,stdv)

    def forward(self, hidden,encoder_outputs):
        '''
        :param hidden:   (batch,hidden_size)
        :param encoder_outputs: (x_seq,batch,hidden_size)
        :return:
        '''
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep,1,1).transpose(0,1)         #(batch,x_seq,hiden_size)
        encoder_outputs = encoder_outputs.transpose(0,1)

        atten_energies = self.score(h,encoder_outputs)  #(batch,x_seq)
        return F.relu(atten_energies).unsqueeze(1)  #(batch,1,x_seq)

    def score(self,hidden,encoder_outputs):
        energy = F.softmax(self.attn(torch.cat([hidden,encoder_outputs],2)))
        energy = energy.transpose(1,2) #(batch,hidden_size,x_seq)
        v = self.v.repeat(encoder_outputs.size(0),1).unsqueeze(1) #(batch,1,hidden_size)
        energy = torch.bmm(v,energy)  #(batch,1,x_seq)
        return energy.squeeze(1)

class Decoder(nn.Module):
    def __init__(self,embed_size,hidden_size,out_dim,n_layers = 1,dropout=0.2):
        super(Decoder,self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size+out_dim,hidden_size,n_layers,dropout=dropout)
        self.out = nn.Linear(hidden_size*2,out_dim)

    def forward(self, x_in,last_hidden,encoder_outputs):
        '''
        :param x_in: (1,batch,embed_size)
        :param last_hidden:  (n_layers,batch,hidden_size)
        :param encoder_outputs:  (x_seq,batch,hidden_size)
        :return:
        '''

        # start_flag = torch.zeros((1,encoder_outputs.shape[1],self.embed_size))

        #Calculate attention weights and apply to encoder outputs

        #atten_weights: (batch,1,x_seq)
        atten_weights = self.attention(last_hidden[-1],encoder_outputs)

        #(batch,1,x_seq) * ( batch,x_seq,hidden_size)
        context = atten_weights.bmm(encoder_outputs.transpose(0,1))  #(batch,1,hidden_size)

        context = context.transpose(0,1)  #(1,bacth,hidden_size)

        #Combine embedded input word and attended context,run through RNN
        # print(x_in.size(),context.size())
        rnn_input = torch.cat([x_in,context],dim=2)
        output,hidden = self.gru(rnn_input,last_hidden)  #output:(1,batch,hidden_size)

        output = output.squeeze(0)
        context = context.squeeze(0)

        out = self.out(torch.cat([output,context],1))    #out: (batch,out_dim)
        out = out.unsqueeze(0)
        return out,hidden,atten_weights

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,seed):
        super(Seq2Seq,self).__init__()

        torch.manual_seed(seed)
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x_train,y_len):
        '''
        :param x_train: (x_seq,bacth,embed_size)
        :param y_train: (y_len,batch,out_dim)
        :return:
        '''
        batch_size = x_train.shape[1]
        max_len = y_len
        out_dim = self.decoder.out_dim
        outputs = Variable(torch.zeros(max_len,batch_size,out_dim))

        encoder_output,hidden = self.encoder(x_train)
        output = torch.zeros((1,batch_size,out_dim))

        for i in range(max_len):
            output,hidden,attn_weights = self.decoder(output,hidden,encoder_output)

            outputs[i] = output

        return outputs

