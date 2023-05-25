import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)    # (input_features, embedding_size, num_layers)
		# self.rnn = CustomLSTM(hidden_size, num_layers)    # (input_features, embedding_size, num_layers)
	
	def forward(self, x):#, state):   # x : integer encoding (128,20)
		""" TO DO: feed the unpacked input x to Encoder """
        
		emb = self.embedding(x)      	# trainable_embedding (128,20,512)
        
		# xt = x.transpose(0,1) # (20,128)
		inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)   # length_list
                
		packed = pack(emb, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
		# embt = emb.permute(1,0,2)		# torch.Size([20, 128, 512])
		output, state = self.rnn(packed)#, state)
		output, outputs_length = unpack(output, total_length=x.shape[1])  # (128,20,512)
		# print("a")
		# print(output, output.shape)

		return output, state
	

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		# self.rnn = nn.LSTM(hidden_size,hidden_size, num_layers)
		self.rnn = CustomLSTM(hidden_size, num_layers)
		self.fc_out = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=-1)
		)
	
	def forward(self, x, state):
		""" TO DO: feed the input x to Decoder """
		x = x.unsqueeze(0)
		emb = self.embedding(x)
		output, state = self.rnn(emb, state)
		prediction = self.fc_out(output.squeeze(0))
	
		return prediction, output, state


class Attention(nn.Module):
    def __init__(self, hidden_size, vocab_size, **kwargs):
        super(Attention, self).__init__()
        self.wc = nn.Linear(hidden_size * 2, hidden_size) # (embed_size * 2, embed_size) = (8, 4)
        self.tanh = nn.Tanh()
        self.wy = nn.Linear(hidden_size, vocab_size) # (embed_size, word_cnt)
    def forward(self, x):
        # (1,1,embed_size * 2)
        x = self.wc(x)
        # (1,1,embed_size)
        x = self.tanh(x)
        # (1,1,embed_size)
        x = self.wy(x)
        # (1,1,word_cnt)
        x = F.log_softmax(x, dim=2)
        # (1,1,word_cnt)
        return x
    

class CustomLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTMCell을 num_layers만큼 생성
        self.cells = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input, initial_states=None):
        batch_size = input.size(1)
        seq_length = input.size(0)

        # 초기 은닉 상태와 셀 상태를 초기화
        if initial_states is None:
            h_t = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(self.num_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(self.num_layers)]
        else:
            h_t, c_t = initial_states

        # 각 타임스텝에 대해 LSTMCell을 적용
        outputs = []
        for t in range(seq_length):
            x_t = input[t, :, :]
            layer_h_t = []
            layer_c_t = []
            for layer in range(self.num_layers):
                h, c = self.cells[layer](x_t, (h_t[layer], c_t[layer]))
                x_t = h
                layer_h_t.append(h)
                layer_c_t.append(c)
            outputs.append(x_t)
            h_t = layer_h_t
            c_t = layer_c_t

        # 출력과 각 레이어의 마지막 상태(state)를 시퀀스 형태로 변환하여 반환
        outputs = torch.stack(outputs, dim=0)

        states_h = torch.stack(layer_h_t, dim=0)
        states_c = torch.stack(layer_c_t, dim=0)

        return outputs, (states_h, states_c)
    



# import numpy as np

# def pack_padded_sequence(input, lengths, batch_first=False):
#     if batch_first:
#         input = np.transpose(input)
    
#     sorted_indices = np.argsort(lengths)[::-1]
#     sorted_input = input[sorted_indices]
#     sorted_lengths = lengths[sorted_indices]
    
#     packed_input = (sorted_input, sorted_lengths)
    
#     return packed_input

# # 사용 예시
# input = np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0], [7, 8, 9]])
# lengths = np.array([3, 2, 1, 3])

# packed_input = pack_padded_sequence(input, lengths, batch_first=True)
# print(packed_input)
