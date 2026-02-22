import torch
import torch.nn as nn
from Model_CFMTDC.embeddingModule import EmbeddingLayer

class Predictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layer, dropout, maxLen, bias=False):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size,
                                              embed_dim=embedding_dim,
                                              max_len=maxLen)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layer, bias=bias, batch_first=True, dropout=dropout if num_layer > 1 else 0)
        
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, y, hidden=None):
        """_summary_
        Args:
            y (torch tensor): [Batchsize, seqlen]
            hidden (tuple(h, c)): from previous step
        Returns:
            gu: [B, U, hidden_dim]
            (h, c) new hidden state
        """
        y_embedd = self.embedding_layer(y)
        lstm_out
        return 