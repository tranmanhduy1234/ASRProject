import torch
import torch.nn as nn
import math
import config

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout=0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD) 
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
        self.max_len = max_len
    def forward(self, x):
        seq_len = x.size(1)
        assert seq_len <= self.max_len, f"Độ dài chuỗi đầu vào phải nhỏ hơn Maxlen: {self.max_len}"
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embed(x) * self.scale + self.pos_embed(positions)
        return self.dropout(x)