import torch
import torch.nn as nn

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ffn, dropout, bias):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.bias = bias
        
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.SiLU()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn, bias=self.bias)
        self.linear2 = nn.Linear(d_ffn, d_model, bias=self.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.dropout(self.activate(x))
        x = self.dropout(self.linear2(x))
        return 0.5 * x + residual

if __name__=="__main__":
    x = torch.randn(2, 13, 25)
    module = FeedForwardModule(25, 50, 0.1, bias=False)
    out = module(x)
    print(out.shape)