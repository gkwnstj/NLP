import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack



class Encoder(nn.Module):
    def __init__(self, CustomLSTM ,vocab_size, hidden_size, num_layers=4, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)    # (input_features, embedding_size, num_layers)
        self.rnn = CustomLSTM    # (input_features, embedding_size, num_layers)

    def forward(self, x):#, state):   # x : integer encoding (128,20)
        """ TO DO: feed the unpacked input x to Encoder """

        # xt = x.transpose(0,1) # (20,128)
        inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1)   # length_list
        emb = self.embedding(x)      	# trainable_embedding (20,128,512)
        # packed = pack(emb, inputs_length.tolist(), enforce_sorted=False)
        # embt = emb.permute(1,0,2)		# torch.Size([20, 128, 512])
        packed = pack(emb, inputs_length.tolist(), batch_first=True, enforce_sorted=False)   # (all_sentence_len, embedding_size)
        start = 0
        outputlist = []
        statelist = torch.zeros(128, 20, self.hidden_size)
        # hlist = []
        # clist = []
        t = 0
        for length in packed.batch_sizes:
            end = start + length
            data = packed.data[start:end]   # (batch_size(ex)fifth_words)), hidden_size)(5,512) 
            data = data.unsqueeze(0)   # (1,65,512)   # we pad from 65 to 128
            output, state = self.rnn(data) # torch.Size([1, 65, 512])
            padding = max(0, 128 - data.shape[1])
            output = F.pad(output, (0, 0, 0, padding, 0, 0), value=0)
            # output = F.pad(state[0], (0, 0, 0, padding, 0, 0), value=0)
            # output = F.pad(state[1], (0, 0, 0, padding, 0, 0), value=0)
            outputlist.append(output)
            # hlist.append(state[0])
            # clist.append(state[1])
            t =+1
        h = F.pad(state[0], (0, 0, 0, padding, 0, 0), value=0)
        c = F.pad(state[1], (0, 0, 0, padding, 0, 0), value=0)
        state = (h,c)
        output = torch.stack(outputlist).squeeze(1)
        output = output[:, packed.unsorted_indices,:]
        # output, state = self.rnn(embt)#, state)
        # output, outputs_length = unpack(output, total_length=x.shape[1])  # (128,20,512)

        # print(output, output.shape)
        return output, state
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        return F.softmax(attention, dim = 1)
	

class Decoder(nn.Module):
    def __init__(self, CustomLSTM, attention, vocab_size, hidden_size, num_layers=4, **kwargs):
        super(Decoder, self).__init__()
        self.attention = attention
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        """ TO DO: Implement your LSTM """
        # self.rnn = nn.LSTM(hidden_size*2,hidden_size, num_layers)
        self.rnn = CustomLSTM
        self.fc_out = nn.Linear(hidden_size*3, vocab_size)

     
    def forward(self, x, state, encoder_outputs):
        """ TO DO: feed the input x to Decoder """
        hidden = state[0]   
        x = x.unsqueeze(0)   # # x [1, 128]
        embedded = self.embedding(x)  # torch.Size([128, 1, 20])

        a = self.attention(hidden, encoder_outputs)   # attention value (128, 20)

        
        # h = encoder_outputs.permute(1,0,2)  # (128,20,512)
        # b = a.unsqueeze(2)
        # att_v = torch.sum(h * b, dim=1).unsqueeze(1)   #expected(128,1,512)
        # con = torch.cat((att_v, outputs.permute(1,0,2)), dim = 2) # (128,1,1024)
        # out = attention(con)      # (128,1,25000)
        # outputs_list[i] = out.squeeze()   # (128,25000)

        a = a.unsqueeze(1)  #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) #torch.Size([128, 20, 512])
        weighted = torch.bmm(a, encoder_outputs)       # convext vector
        weighted = weighted.permute(1, 0, 2)    # torch.Size([1, 128, 512])

        rnn_input = torch.cat((embedded, weighted), dim = 2)  # torch.Size([1, 128, 1024])
        output, hidden = self.rnn(rnn_input, state)
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden#.squeeze(0), a.squeeze(1)


# class CustomLSTM(nn.Module):
#     def __init__(self, hidden_size, out_him, num_layers):
#         super(CustomLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.out_hidden = out_him
#         self.num_layers = num_layers
        
#         # LSTMCell을 num_layers만큼 생성
#         self.cells = nn.ModuleList([nn.LSTMCell(hidden_size, out_him) for _ in range(num_layers)])

#     def forward(self, input, initial_states=None):
#         batch_size = input.size(1)
#         seq_length = input.size(0)

#         # 초기 은닉 상태와 셀 상태를 초기화
#         if initial_states is None:
#             h_t = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(self.num_layers)]
#             c_t = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(self.num_layers)]
#         else:
#             h_t, c_t = initial_states

#         # 각 타임스텝에 대해 LSTMCell을 적용
#         outputs = []
#         for t in range(seq_length):
#             x_t = input[t, :, :]
#             layer_h_t = []
#             layer_c_t = []
#             for layer in range(self.num_layers):
#                 h, c = self.cells[layer](x_t, (h_t[layer], c_t[layer]))
#                 x_t = h
#                 layer_h_t.append(h)
#                 layer_c_t.append(c)
#             outputs.append(x_t)
#             h_t = layer_h_t
#             c_t = layer_c_t

#         # 출력과 각 레이어의 마지막 상태(state)를 시퀀스 형태로 변환하여 반환
#         outputs = torch.stack(outputs, dim=0)

#         states_h = torch.stack(layer_h_t, dim=0)
#         states_c = torch.stack(layer_c_t, dim=0)

#         return outputs, (states_h, states_c)
    

class CustomLSTM(nn.Module):
    def __init__(self, hidden_size, out_hidden, num_layers):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.out_hidden = out_hidden
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        
        # 첫 번째 레이어는 다른 크기를 가짐
        self.cells.append(nn.LSTMCell(hidden_size, out_hidden))
        
        # 두 번째부터 마지막 레이어까지는 동일한 크기를 가짐
        for _ in range(num_layers - 1):
            self.cells.append(nn.LSTMCell(out_hidden, out_hidden))

    def forward(self, input, initial_states=None):
        batch_size = input.size(1)
        seq_length = input.size(0)

        if initial_states is None:
            h_t = [torch.zeros(batch_size, self.hidden_size).to(input.device)]
            c_t = [torch.zeros(batch_size, self.hidden_size).to(input.device)]
            
            # 두 번째 레이어부터 초기 상태 초기화
            for _ in range(self.num_layers - 1):
                h_t.append(torch.zeros(batch_size, self.out_hidden).to(input.device))
                c_t.append(torch.zeros(batch_size, self.out_hidden).to(input.device))
        else:
            h_t, c_t = initial_states

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
