import torch
import torch.nn as nn
from Model_CFMTDC.feedforwardmodule import FeedForwardModule
from Model_CFMTDC.r_mhsa import RelativePositionalMultiHeadSelfAttention
from Model_CFMTDC.convolution_module import ConformerConvModule

class ConformerBlockEncoder(nn.Module):
    def __init__(self, d_model, d_ffn, num_head, expansion_factor, kernel_size, dropout, bias):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model=d_model, d_ffn=d_ffn, dropout=dropout, bias=bias)
        self.msha = RelativePositionalMultiHeadSelfAttention(d_model=d_model, n_head=num_head, dropout=dropout, bias=bias)
        self.convmodule = ConformerConvModule(d_model=d_model, expansion_factor=expansion_factor, 
                                              kernel_size=kernel_size, dropout_p=dropout, bias=bias)
        self.ffn2 = FeedForwardModule(d_model=d_model, d_ffn=d_ffn, dropout=dropout, bias=bias)
        self.norm = nn.LayerNorm(d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.ffn1(x)
        
        residual = x
        x = self.norm(x)
        x = self.dropout(self.msha(x, x, x, mask))
        x = x + residual
        
        x = self.convmodule(x, mask)
        return self.norm(self.ffn2(x))