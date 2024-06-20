import torch
import torch.nn as nn
import math

# PAPER https://arxiv.org/abs/1706.03762

class InputEmbeddings(nn.Module):
    """
    Embedding is a means of representing objects like text, images and audio as points in a continuous vector space 
    where the locations of those points in space are semantically meaningful.

    This class transform the original sentence into a list of vectors which then it's learn by the model
    """
    def __init___(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.voacab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    """
    Positional Encoding is only computed once and reused for every sentence during training and inference
    """
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (len_q)
        position =  torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float()* (-math.log(10000.0) / d_model))
        # apply the sin in every even position
        pe[:,0::2] = torch.sin(position*div_term)
        # apply cos in every odd position
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1], :]).requieres_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean)/(std+self.eps)+ self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



# Following video https://www.youtube.com/watch?v=ISNdQcPhsts&t=1194s
# at 24:00