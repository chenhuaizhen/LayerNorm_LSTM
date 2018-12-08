import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True, use_layer_norm=True):
        super().__init__(input_size, hidden_size, bias)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
            self.ln_ho = nn.LayerNorm(hidden_size)
        # DropConnect on the recurrent hidden to hidden weight
        self.dropout = dropout

    def forward(self, input, hidden=None):
        self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden
        self.check_forward_hidden(input, hx, '[0]')
        self.check_forward_hidden(input, cx, '[1]')

        weight_hh = nn.functional.dropout(self.weight_hh, p=self.dropout, training=self.training)
        if self.use_layer_norm:
            gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                     + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))
        else:
            gates = F.linear(input, self.weight_ih, self.bias_ih) \
                    + F.linear(hx, weight_hh, self.bias_hh)

        i, f, c, o = gates.chunk(4, 1)
        i_ = torch.sigmoid(i)
        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c)
        o_ = torch.sigmoid(o)
        cy = (f_ * cx) + (i_ * c_)
        if self.use_layer_norm:
            hy = o_ * self.ln_ho(torch.tanh(cy))
        else:
            hy = o_ * torch.tanh(cy)
        return hy, cy


class LayerNormLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 weight_dropout=0.0,
                 bias=True,
                 bidirectional=False,
                 use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # using variational dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
                for layer in range(num_layers)
            ])

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.hidden1[layer].register_parameter(sub_name, param[1])
            else:
                self.hidden0[layer].register_parameter(sub_name, param[1])

    def forward(self, input, hidden=None, seq_lens=None):
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_size, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        if self.bidirectional:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_ = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_reverse = torch.LongTensor([0] * batch_size).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            indices = torch.cat((indices_, indices_reverse), dim=1)
            hy = []
            cy = []
            xs = input
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, 2, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, 2, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht_, ct_ = layer0(x0, (h0, c0))
                    ht[t][l0] = ht_ * seq_len_mask[t]
                    ct[t][l0] = ct_ * seq_len_mask[t]
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht_, ct_ = layer1(x1, (h1, c1))
                    ht[t][l1] = ht_ * seq_len_mask[t]
                    ct[t][l1] = ct_ * seq_len_mask[t]
                    h1, c1 = ht[t][l1], ct[t][l1]

                xs = [torch.cat((h[l0]*dropout_mask[l][0], h[l1]*dropout_mask[l][1]), dim=1) for h in ht]
                ht_temp = torch.stack([torch.stack([h[l0], h[l1]]) for h in ht])
                ct_temp = torch.stack([torch.stack([c[l0], c[l1]]) for c in ct])
                if len(hy) == 0:
                    hy = torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    hy = torch.cat((hy, torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
                if len(cy) == 0:
                    cy = torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    cy = torch.cat((cy, torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
            y  = torch.stack(xs)
        else:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices = (torch.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, self.num_layers, 1, self.hidden_size])
            h, c = hx, cx
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht_, ct_ = layer(x, (h[l], c[l]))
                    ht[t][l] = ht_ * seq_len_mask[t]
                    ct[t][l] = ct_ * seq_len_mask[t]
                    x = ht[t][l] * dropout_mask[l]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1]*dropout_mask[-1] for h in ht])
            hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0)))
            cy = torch.stack(list(torch.stack(ct).gather(dim=0, index=indices).squeeze(0)))

        return y, (hy, cy)

'''
test the module
'''
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
def is_equal(a, b, epsilon=1e-5):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), epsilon)).item() == 1

def test_layernorm_LSTMCell():
    batch_size = 4
    hidden_size = 2
    num_input_features = 3
    # create two objects
    rnn = LayerNormLSTMCell(num_input_features, hidden_size, bias=True, use_layer_norm=False)
    rnn_old = torch.nn.LSTMCell(num_input_features, hidden_size, bias=True)
    # initialize two objects with same weights & biases
    for param in rnn_old.named_parameters():
        rnn.register_parameter(param[0], param[1])
    # initialize the hidden state
    states = (torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size))
    # create the input data
    input_tensor = torch.FloatTensor(np.random.rand(batch_size, num_input_features))

    # normal operation for use LSTM to decode the data
    rnn_old_h, rnn_old_c = rnn_old(input_tensor, states)
    # use the new LSTM to decode the data
    rnn_h, rnn_c = rnn(input_tensor, states)

    # check whether the two objects' outputs are the same
    print("whether the two objects' h_1 are the same: ", is_equal(rnn_old_h, rnn_h))
    print("whether the two objects' c_1 are the same: ", is_equal(rnn_old_c, rnn_c))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size)
    f = torch.matmul(rnn_h, x)
    f.backward(torch.ones(batch_size))
    print("the backward operation can be run normally")

def test_layernorm_LSTM(use_biLSTM=True):
    batch_size = 4
    max_length = 3
    hidden_size = 2
    n_layer = 5
    num_input_features = 3
    n_direction = 2 if use_biLSTM else 1
    # create two objects
    rnn = LayerNormLSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM, use_layer_norm=False)
    rnn_old = torch.nn.LSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM)
    # initialize two objects with same weights
    rnn.copy_parameters(rnn_old)
    # initialize the hidden state
    states = (torch.zeros(n_layer*n_direction, batch_size, hidden_size), torch.zeros(n_layer*n_direction, batch_size, hidden_size))
    # create the sequence data with padding
    input_tensor = torch.zeros(batch_size, max_length, num_input_features)
    input_tensor[0] = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    input_tensor[1] = torch.FloatTensor([[4, 5, 6], [5, 7, 8], [0, 0, 0]])
    input_tensor[2] = torch.FloatTensor([[6, 4, 3], [8, 1, 9], [0, 0, 0]])
    input_tensor[3] = torch.FloatTensor([[7, 3, 5], [0, 0, 0], [0, 0, 0]])
    seq_lengths = [3, 2, 2, 1]
    # transform the sequence data into new shape [max_length, batch_size, num_input_features]
    batch_in = Variable(input_tensor)
    batch_in = batch_in.permute(1, 0, 2)
    # normal operation for use LSTM to decode the sequence
    pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths)
    rnn_old_out, rnn_old_states = rnn_old(pack, states)
    rnn_old_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_old_out)
    # use the new LSTM to decode the sequence
    rnn_out, rnn_states = rnn(batch_in, states, seq_lengths)

    # check whether the two objects' outputs are the same
    print("whether the two objects' outputs are the same: ", is_equal(rnn_old_out, rnn_out))
    print("whether the two objects' h_n are the same: ", is_equal(rnn_old_states[0], rnn_states[0]))
    print("whether the two objects' c_n are the same: ", is_equal(rnn_old_states[1], rnn_states[1]))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size * n_direction)
    f = torch.matmul(rnn_out, x)
    f.backward(torch.ones(max_length, batch_size))
    print("the backward operation can be run normally")


if __name__ == "__main__":
    print("start checking the layernorm-LSTMCell......")
    test_layernorm_LSTMCell()
    print()
    print("start checking the layernorm-LSTM......")
    test_layernorm_LSTM(use_biLSTM=False)
    print()
    print("start checking the bi-layernorm-LSTM......")
    test_layernorm_LSTM(use_biLSTM=True)
