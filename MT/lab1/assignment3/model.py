import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout) #Dropout function
        self.encoder = nn.Embedding(ntoken, ninp) #encoder
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout) #instantiate RNN model (recurrent module)
        self.decoder = nn.Linear(nhid, ntoken) #instantiate decoder (linear layer)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        """TODO: complement the forward computation,                                   
        # given the input and the hidden states
        # return softmax results and the hidden statesm
        # hints: rnn -> dropout -> output layer -> log_softmax
        rnn takes the droput input as parameter  """
        emb = self.drop(self.encoder(input)) #applies dropout to input

        rnn_out, hn = self.rnn(emb, hidden) #feeds dropout-ed input and latest hidden state to the rnn, returns last layer output and hidden state
        #rnn_out shape is 35,20,25. 25 is the vocabulary value
        
        logits = self.decoder(rnn_out) #decoder/linear layer, gives logits

        softmax = nn.LogSoftmax(dim=2) #create softmax layer, LogSoftmax to match the loss function in nlm
        
        normalized = softmax(logits) #normalize the output layer with softmax

        normalized = normalized.view(-1, normalized.size(2)) #targets is size 700, -1 gives rnn_out 0*1 (35,20) = 700
        #Size(2) keeps the vocabulary size
        return normalized, hn #return the normalized, reshaped values and the hidden states
        

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)




    


