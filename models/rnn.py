import torch.nn as nn


class RNNClassifier(nn.Module):
    
    def __init__(self, input_size, word_vec_size, hidden_size, 
                 n_classes, n_layers=4, dropout_p=.3):
        
        self.inputSize = input_size
        self.wordVecSize = word_vec_size
        self.hiddenSize = hidden_size
        self.nClasses = n_classes
        self.nLayers = n_layers
        self.dropoutP = dropout_p
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size, 
            hidden_size=hidden_size, 
            num_layers=n_layers, 
            dropout=dropout_p, 
            batch_first=True, 
            bidirectional=True, 
        )
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        self.activation = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        
        return y
    