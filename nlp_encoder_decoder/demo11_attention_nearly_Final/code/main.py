import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils, dataloader, lstm_example
import pandas as pd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

def main(args):

	utils.set_random_seed(seed_num=args.seed_num)

	use_cuda = utils.check_gpu_id(args.gpu_id)
	device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')
	print("Using_{}_device".format(device))

	t_start = time.time()

	vocab_src = utils.read_pkl('./data/de-en/nmt_simple.src.vocab.pkl')
	# vocab_src.update({'<SOS>' : 35819})
	vocab_tgt = utils.read_pkl('./data/de-en/nmt_simple.tgt.vocab.pkl')
	# vocab_tgt.update({'<SOS>' : 24999})

	tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											src_filepath='./data/de-en/nmt_simple.src.train.txt',
											tgt_filepath='./data/de-en/nmt_simple.tgt.train.txt',
											vocab=(vocab_src, vocab_tgt))
	ts_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
											src_filepath='./data/de-en/nmt_simple.src.test.txt',
											vocab=(tr_dataset.vocab_src, tr_dataset.vocab_tgt))
	vocab_src = tr_dataset.vocab_src
	vocab_tgt = tr_dataset.vocab_tgt
	i2w_src = {v:k for k, v in vocab_src.items()}
	i2w_tgt = {v:k for k, v in vocab_tgt.items()}

	tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
	ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

	encoder = lstm_example.Encoder(len(vocab_src), args.hidden_size, num_layers=args.num_layers)

	decoder = lstm_example.Decoder(len(vocab_tgt), args.hidden_size, num_layers=args.num_layers)

	attention = lstm_example.Attention(args.hidden_size, len(vocab_tgt))

	utils.init_weights(encoder, init_type='uniform_')
	utils.init_weights(decoder, init_type='uniform_')
	utils.init_weights(attention, init_type='uniform_')
	
	encoder = encoder.to(device)
	decoder = decoder.to(device)
	attention = attention.to(device)

	""" TO DO: (masking) convert this line for masking [PAD] token """
	criterion = nn.NLLLoss(ignore_index=0)

	optimizer_enc = optim.Adam(encoder.parameters(), lr=args.lr)
	optimizer_dec = optim.Adam(decoder.parameters(), lr=args.lr)
	optimizer_attn = optim.Adam(attention.parameters(), lr=args.lr)


	def train(dataloader, epoch, tgt_vocab_size, teacher_forcing, autoregressive):
		encoder.train()
		decoder.train()
		attention.train()

		tr_loss = 0.
		correct = 0

		cnt = 0
		total_score = 0.
		prev_time = time.time()
		for idx, (src, tgt) in enumerate(dataloader):
			src, tgt = src.to(device), tgt.to(device)

			optimizer_enc.zero_grad()
			optimizer_dec.zero_grad()
			optimizer_attn.zero_grad()

			
			enc_outputs, (h,c) = encoder(src)     # context vector enc_outputs : (20,128,512) trained encoding, src : (128,20) integer encoding, 
			h1 = h		# torch.Size([4, 128, 512])
			c1 = c		# torch.Size([4, 128, 512])


			tgt_decinput = tgt  # torch.Size([128, 20]) # tgt_decinput includes <eos> which is end of sentence

			# # 첫 번째 열에 2999를 추가 <SOS>
			# tgt_tensor = torch.cat((torch.full((args.batch_size, 1), 2999, dtype=torch.int64).to(device), tgt_decinput), dim=1).to(device)

			# # 마지막 열 제거
			# tgt_tensor = tgt_tensor[:, :-1]

			# # 행의 마지막 열에 2가 있는 경우 해당 값을 0으로 변경
			# tgt_tensor[tgt_tensor.eq(2)] = 0

			# tgt_input = tgt_tensor.permute(1,0)
			# dec_in = tgt_input[0,:]     # <SOS>


			outputs_list = torch.zeros(args.max_len, args.batch_size, tgt_vocab_size).to(device)
			dec_in = torch.zeros(args.batch_size, dtype = torch.long).to(device)
			# tgt[:] = 2999
			dec_in[:] = 2
			nonauto = dec_in		# torch.Size([128])
			state = (h1,c1)			# h1 : torch.Size([4, 128, 512]), c1 : torch.Size([4, 128, 512])
			for i in range(args.max_len):
				if not args.autoregressive:
					dec_in = nonauto
				pred, outputs ,state = decoder(dec_in, state)   # <EOS>  ([128])
				score_list = []
				for j in range(args.max_len):
					enc_h = enc_outputs[j].unsqueeze(0).permute(1,0,2)   # (128,1,512)
					dec_h = outputs.permute(1,2,0)						 # (128,512,1)
					score = torch.bmm(enc_h, dec_h).squeeze()			 # torch.Size([128])
					score_list.append(score)					# (20)
				# score = torch.bmm(enc_outputs, outputs.permute(1,2,0)).squeeze()

				stacked_score = torch.stack(score_list,dim=1).to(device)   # torch.Size([128, 20])
				att_dis = F.softmax(stacked_score, dim=1)		# torch.Size([128, 20])
				h = enc_outputs.permute(1,0,2)   # (128,20,512)
				b = att_dis.unsqueeze(2)
				att_v = torch.sum(h * b, dim=1).unsqueeze(1)   #expected(128,1,512)
				con = torch.cat((att_v, outputs.permute(1,0,2)), dim = 2) # (128,1,1024)
				out = attention(con)      # (128,1,25000)
				outputs_list[i] = out.squeeze()   # (128,25000)
				
				if teacher_forcing:
					dec_in = tgt_decinput.permute(1,0)[i]
				else:
					top1 = out.squeeze().argmax(dim=-1)
			



			outputs_list = outputs_list.permute(1,0,2)   # batch 기준 정렬
			outputs_list = outputs_list.reshape(args.batch_size * args.max_len, -1)  # 문장리스트 한 줄로 정렬 (2560,25000)
			tgt = tgt.reshape(-1)

			loss = criterion(outputs_list, tgt) # output : (5120,512), tgt : (5120)  # output(pred) : start with word, tgt(label) : start with <sos>
			tr_loss += loss.item()
			loss.backward()

			""" TO DO: (clipping) convert this line for clipping the 'gradient < args.max_norm' """
			torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_norm)
			torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_norm)
			torch.nn.utils.clip_grad_norm_(attention.parameters(), args.max_norm)

			optimizer_enc.step()
			optimizer_dec.step()
			optimizer_attn.step()

			# accuracy
			pred = outputs_list.argmax(dim=1, keepdim=True)
			pred_acc = pred[tgt != 0]
			tgt_acc = tgt[tgt != 0]
			correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

			cnt += tgt_acc.shape[0]

			# BLEU score
			score = 0.
			with torch.no_grad():
				pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
				tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
				for p, t in zip(pred, tgt):
					eos_idx = t.index(vocab_tgt['[PAD]']) if vocab_tgt['[PAD]'] in t else len(t)
					p_seq = [i2w_tgt[i[0]] for i in p][:eos_idx]
					t_seq = [i2w_tgt[i] for i in t][:eos_idx]
					k = args.k if len(t_seq) > args.k else len(t_seq)
					s = utils.bleu_score(p_seq, t_seq, k=k)
					score += s
					total_score += s

			score /= args.batch_size

			# verbose
			batches_done = (epoch - 1) * len(dataloader) + idx
			batches_left = args.n_epochs * len(dataloader) - batches_done
			time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
			prev_time = time.time()
			print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
				epoch, args.n_epochs, idx+1, len(dataloader), loss, time_left), end=' ')

		tr_loss /= cnt
		tr_acc = correct / cnt
		tr_score = total_score / len(dataloader.dataset)
		
		return tr_loss, tr_acc, tr_score


	def test(dataloader, tgt_vocab_size, lengths=None):
		encoder.eval()
		decoder.eval()
		attention.eval()

		idx = 0
		total_pred = []

		prev_time = time.time()
		with torch.no_grad():
			for src, _ in dataloader:
				src = src.to(device)

				enc_outputs, (h,c) = encoder(src)
				h1 = h
				c1 = c

				tgt = torch.zeros(args.batch_size, dtype = torch.long).to(device)
				# tgt[:] = 2999
				tgt[:] = 2
				dec_in = tgt
				outputs_list = torch.zeros(args.max_len, args.batch_size, tgt_vocab_size).to(device)
				test_states = (h1,c1)
				for i in range(1,args.max_len):
					pred, outputs ,test_states = decoder(dec_in, test_states)
					score_list = []
					for j in range(args.max_len):
						enc_h = enc_outputs[j].unsqueeze(0).permute(1,0,2)   # (128,1,512)
						dec_h = outputs.permute(1,2,0)						 # (128,512,1)
						score = torch.bmm(enc_h, dec_h).squeeze()
						score_list.append(score)
					stacked_score = torch.stack(score_list,dim=1).to(device)
					att_dis = F.softmax(stacked_score, dim=1)
					h = enc_outputs.permute(1,0,2)   # (128,20,512)
					b = att_dis.unsqueeze(2)
					att_v = torch.sum(h * b, dim=1).unsqueeze(1)   #expected(128,1,512)
					con = torch.cat((att_v, outputs.permute(1,0,2)), dim = 2) # (128,1,1024)
					out = attention(con)      # (128,1,25000)
					outputs_list[i] = out.squeeze()   # (128,25000)
					
					top1 = out.squeeze().argmax(dim=-1)
					dec_in = top1       

				outputs_list = outputs_list.permute(1,0,2)
				# outputs_list = outputs_list.reshape(args.batch_size * args.max_len, -1)

				for k in range(outputs_list.shape[0]):		# outputs.shape[0] : 128
					pred = outputs_list[k].argmax(dim=-1)   # pred : (128) outputs[i] : (128,24999)
					total_pred.append(pred[:lengths[idx+k]].detach().cpu().numpy())   
		
				idx += args.batch_size
		total_pred = np.concatenate(total_pred)


		return total_pred


	for epoch in range(1, args.n_epochs + 1):
		tr_loss, tr_acc, tr_score = train(tr_dataloader, epoch, len(vocab_tgt), args.teacher_forcing, args.autoregressive)
		# {format: (loss, acc, BLEU)}
		print("tr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100), end='')

	print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

	# for kaggle: by using 'pred_{}.npy' to make your submission file
	with open('./data/de-en/nmt_simple_len.tgt.test.npy', 'rb') as f:
		lengths = np.load(f)
	pred = test(ts_dataloader, len(vocab_tgt), lengths=lengths)
	pred_filepath = Path(args.res_dir) / 'pred_{}.npy'.format(args.res_tag)
	np.save(pred_filepath, np.array(pred))


	predicted_labels = pred

	index_list = []
	for i in range(0,len(predicted_labels)):
		index_list.append(f"S{i+1:05d}")
		
	prediction = pd.DataFrame(columns=['id', 'pred'])

	prediction["id"] = index_list
	prediction["pred"] = predicted_labels

	prediction = prediction.reset_index(drop=True)

	prediction.to_csv('./result/20221119_하준서_sent_class.pred.csv', index = False)



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='NMT - Seq2Seq with Attention')
	""" recommend to use default settings """
	# environmental settings
	parser.add_argument('--gpu-id', type=int, default=0)
	parser.add_argument('--seed-num', type=int, default=0)
	parser.add_argument('--save', action='store_true', default=0)
	parser.add_argument('--res-dir', default='./result', type=str)
	parser.add_argument('--res-tag', default='seq2seq', type=str)
	# architecture
	parser.add_argument('--num_layers', type=int, default=4)
	parser.add_argument('--max-len', type=int, default=20)
	parser.add_argument('--hidden-size', type=int, default=512)   # 512
	parser.add_argument('--max-norm', type=float, default=5.0)
	# hyper-parameters
	parser.add_argument('--n_epochs', type=int, default=1)        # 100
	parser.add_argument('--batch-size', type=int, default=128)    # 128
	parser.add_argument('--lr', type=float, default=0.001)   #0.001
	# option
	parser.add_argument('--autoregressive', action='store_true', default=True)
	parser.add_argument('--teacher-forcing', action='store_true', default=False)
	parser.add_argument('--attn', action='store_true', default=False)
	# etc
	parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

	args = parser.parse_args()

	main(args)