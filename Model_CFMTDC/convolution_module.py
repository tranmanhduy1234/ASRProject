import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, expansion_factor=2, kernel_size=31, dropout_p=0.1, bias=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)

        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * expansion_factor * 2, kernel_size=1
        )
        self.glu = GLU(dim=1)

        self.depthwise_conv = nn.Conv1d(
            d_model * expansion_factor, 
            d_model * expansion_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model * expansion_factor, # Depthwise
            bias=bias
        )

        self.batch_norm = nn.BatchNorm1d(d_model * expansion_factor)
        self.swish = Swish()

        self.pointwise_conv2 = nn.Conv1d(
            d_model * expansion_factor, d_model, kernel_size=1
        )
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        residual = x
        
        x = self.layernorm(x)

        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = x + residual
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x

if __name__ == "__main__":
    B, T, D = 2, 50, 256
    conv_mod = ConformerConvModule(d_model=D)
    
    inputs = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 30:] = False 
    
    outputs = conv_mod(inputs, mask=mask)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    
    pad_check = torch.all(outputs[1, 30:, :] == 0)
    print(f"PAD tokens hoàn toàn bằng 0: {pad_check}")