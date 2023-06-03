import math
import time

import torch
from torch import nn, optim
from torch.optim import Adam

from models.model.transformer import Transformer

from torch.utils.data import DataLoader
import utils, dataloader
import argparse
import numpy as np
from pathlib import Path
import pandas as pd


parser = argparse.ArgumentParser(description='NMT - Transformer')
""" recommend to use default settings """

# environmental settings
parser.add_argument('--seed', type=int, default=0)

# architecture
parser.add_argument('--layers', type=int, default=6, help='Number of layers for each Encoder and Decoder')
parser.add_argument('--max_len', type=int, default=20)
parser.add_argument('--model_dim', type=int, default=512, help='Dimension size of model dimension')
parser.add_argument('--hidden_size', type=int, default=2048, help='Dimension size of hidden states')
parser.add_argument('--n_head', type=int, default=8, help='Number of multi-head Attention')
parser.add_argument('--d_prob', type=float, default=0.1, help='Dropout probability')

# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=5e-9, help='Epsilon hyper-parameter for Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-4)

# etc
parser.add_argument('--res-dir', default='./result', type=str)
parser.add_argument('--res-tag', default='transformer', type=str)
parser.add_argument('--factor', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--clip', type=float, default=1.0)


args = parser.parse_args()

utils.set_random_seed(args)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                    src_filepath='./data/de-en/nmt_simple.src.train.txt',
                                    tgt_filepath='./data/de-en/nmt_simple.tgt.train.txt',
                                    vocab=(None, None),
                                    is_src=True, is_tgt=False, is_train=True)
ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                    src_filepath='./data/de-en/nmt_simple.src.test.txt',
                                    tgt_filepath=None,
                                    vocab=(tr_dataset.vocab, None),
                                    is_src=True, is_tgt=False, is_train=False)

vocab_src = tr_dataset.vocab     # ('[PAD]', 2), ('[UNK]', 3), ('[SOS]', 0), ('[EOS]', 1)
vocab_tgt = ts_dataset.vocab


tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
ts_dataloader = DataLoader(ts_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)



src_pad_idx= vocab_src["[PAD]"]
trg_pad_idx= vocab_src["[PAD]"]
trg_sos_idx= vocab_tgt["[SOS]"]
enc_voc_size = len(vocab_src)
dec_voc_size = len(vocab_tgt)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=args.model_dim,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=args.max_len,
                    ffn_hidden=args.hidden_size,
                    n_head=args.n_head,
                    n_layers=args.layers,
                    drop_prob=args.d_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=args.lr,
                 weight_decay=args.weight_decay,
                 eps=args.eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=args.factor,
                                                 patience=args.patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for epoch in range(args.n_epochs):

        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)    # (128,20), (128,21)    you can see the reason why 21 -> convert_sent2seq

            optimizer.zero_grad()
            output = model(src, trg[:, :-1])   # (128, 20, 54887)
            output_reshape = output.contiguous().view(-1, output.shape[-1])  # (2560,54887)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            print('epoch : ', epoch, 'step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
        

    return epoch_loss / len(iterator)


def test(model, iterator, criterion, lengths):
    model.eval()
    total_pred = []
    idx = 0
    with torch.no_grad():


        for i, (src, trg) in enumerate(iterator):
            src, _ = src.to(device), trg.to(device)   # (128,20), (128,21)

            dec_in = torch.zeros(src.shape[0],1, dtype = torch.long).to(device)
            outputs_list = torch.zeros(20, src.shape[0], 54887).to(device)
            
            for t in range(0,args.max_len):

                output = model(src, dec_in)    # (128,1,54887), (128,1), (8,1,54887)
                outputs_list[t] = output.squeeze(1)     # (8,54887)
                dec_in = output.argmax(dim=-1)

            outputs_list = outputs_list.permute(1,0,2)

            for i in range(outputs_list.shape[0]):		# outputs.shape[0] : 128
                pred = outputs_list[i].argmax(dim=-1)   # pred : (128) outputs[i] : (128,24999)
                total_pred.append(pred[:lengths[idx+i]].detach().cpu().numpy())   

            idx += args.batch_size

    
    total_pred = np.concatenate(total_pred)


    return total_pred


def main():

    tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                                src_filepath='./data/de-en/nmt_simple.src.train.txt',
                                                tgt_filepath='./data/de-en/nmt_simple.tgt.train.txt',
                                                vocab=(None, None),
                                                is_src=True, is_tgt=False, is_train=True)
    ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
                                                src_filepath='./data/de-en/nmt_simple.src.test.txt',
                                                tgt_filepath=None,
                                                vocab=(tr_dataset.vocab, None),
                                                is_src=True, is_tgt=False, is_train=False)


    vocab = tr_dataset.vocab     # ('[PAD]', 2), ('[UNK]', 3), ('[SOS]', 0), ('[EOS]', 1)
    # i2w = {v: k for k, v in vocab.items()}

    tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2)
    ts_dataloader = DataLoader(ts_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)


    with open('./data/de-en/length.npy', 'rb') as f:
        lengths = np.load(f)


    train(model, tr_dataloader, optimizer, criterion, args.clip)
    pred = test(model, ts_dataloader, criterion, lengths)

    pred_filepath = Path(args.res_dir) / 'pred_{}.npy'.format(args.res_tag)
    np.save(pred_filepath, np.array(pred))
    predicted_labels = pred




    predicted_labels = pred

    index_list = []
    for i in range(0,len(predicted_labels)):
        index_list.append(f"S{i:05d}")
        
    prediction = pd.DataFrame(columns=['ID', 'label'])

    prediction["ID"] = index_list
    prediction["label"] = predicted_labels

    prediction = prediction.reset_index(drop=True)

    prediction.to_csv('./result/20221119_하준서_sent_class.pred.csv', index = False)



if __name__ == '__main__':
    main()
