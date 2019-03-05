import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
import os
from argparse import Namespace
from Seq2SeqWithAttention import Encoder,Decoder,Seq2Seq
from utils import dataLoader
from plot_graph import plot_temperature
import numpy as np

args = Namespace(
    epochs = 50,
    batch_size = 32,
    lr = 0.001,
    grad_clip = 10.0,
    embed_size = 55 + 41,
    out_dim = 41,
    hidden_size = 128,
    x_len = 10,
    y_len = 10,
    train_size = 0.98,
    val_size = 0.01,
    seed = 1234
)
def train(model,optimizer,train_set):
    model.train()

    n_batch = len(train_set['X']) // args.batch_size
    train_X = train_set['X'][:n_batch*args.batch_size]
    train_y = train_set['y'][:n_batch*args.batch_size]
    _,x_len,embed_dim = train_X.shape
    _, y_len, out_dim = train_y.shape
    train_X = torch.Tensor(train_X.reshape(n_batch,x_len,args.batch_size,embed_dim))
    train_y = torch.Tensor(train_y.reshape(n_batch,y_len,args.batch_size,out_dim))

    total_loss = 0
    for i, batch in enumerate(range(train_X.shape[0])):

        optimizer.zero_grad()

        output = model(train_X[i], y_len)
        # print(output)
        # print('\n'*5)
        output = output.view(-1,args.out_dim)
        target = train_y[i].view(-1,args.out_dim)

        loss = nn.MSELoss()(output,target)

        loss.backward()

        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if i % 100 == 0 and i != 0:
            total_loss = total_loss / 100
            print("------->[%d][train_loss:%5.2f]" %
                  (i, total_loss))
            total_loss = 0
def evaluate(model,val_set):
    model.eval()

    val_X = val_set['X']
    val_y = val_set['y']
    _, x_len, embed_dim = val_X.shape
    _, y_len, out_dim = val_y.shape
    val_X = torch.Tensor(val_X.reshape(x_len, -1, embed_dim))
    val_y = torch.Tensor(val_y.reshape(y_len, -1, out_dim))

    output = model(val_X, y_len)

    output = output.view(-1,out_dim)
    target = val_y.view(-1,out_dim)

    print(output)
    print(target)

    loss = nn.MSELoss()(output,target)

    return loss
def inference(model,infer_set):

    model.eval()
    # model.load_state_dict(torch.load('./save_model/seq2seq_99.pt'))

    infer = torch.Tensor(infer_set['X'].transpose(1, 0, 2))

    output = model(infer, args.y_len)
    # loss = evaluate(model, infer_set)

    return output

def run(seq2seq,train_set,val_set,test_set):

    optimizer = optim.Adam(seq2seq.parameters(),lr = args.lr)

    best_val_loss = None
    for i in range(1,args.epochs+1):
        print('=====>Epoch:{}'.format(i))
        train(model=seq2seq, optimizer=optimizer, train_set = train_set)

        val_loss = evaluate(model=seq2seq,val_set = val_set)

        print("######val_loss:%5.3f"%(val_loss))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir("save_model6"):
                os.makedirs("save_model6")
            torch.save(seq2seq.state_dict(), './save_model6/seq2seq_%d.pt' % (i))
            best_val_loss = val_loss

    test_loss = evaluate(model=seq2seq, val_set=test_set)
    #
    print("[TEST] loss:%5.2f" % test_loss)

def main():
    print('[!] preparing dataset...')
    data_loader = dataLoader(seed=args.seed)

    train_set, val_set, test_set,infer_set = data_loader.split_dataset(x_len=args.x_len, y_len=args.y_len,
                                                              train_size=args.train_size, val_size=args.val_size)
    print('[!]Installing models...')
    encoder = Encoder(args.embed_size,args.hidden_size,
                      n_layers = 1,dropout = 0.5)
    decoder = Decoder(args.embed_size,args.hidden_size,
                      args.out_dim,n_layers = 1,dropout=0.5)
    seq2seq = Seq2Seq(encoder,decoder,seed=args.seed)
    print(seq2seq)

    # run(seq2seq,train_set,val_set,test_set)
    seq2seq.load_state_dict(torch.load('./save_model6/seq2seq_25.pt'))
    '''
    inference stage:
    '''
    out = inference(seq2seq, infer_set)
    out = out.detach().numpy().squeeze(1)

    new_temperature,mean,std = data_loader.new_temperature,data_loader.temperature_mean,data_loader.temperature_std

    index = 1
    plot_temperature(out[:,index],new_temperature[:,index],mean[index],std[index])

    # print(new_temperature.shape)
    # print(mean.shape)
    # print(std)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('[STOP]',e)