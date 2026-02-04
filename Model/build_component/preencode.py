import torch
import torch.nn as nn
import math

class AudioEncoderEmbedding(nn.Module):
    def __init__(self, n_mels=80, d_model=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, d_model)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, d_model)
        self.gelu2 = nn.GELU()
        
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, d_model)
        self.gelu3 = nn.GELU()
        
        self.d_model = d_model
        self.reset_parameters()
        
    def forward(self, x):
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        
        x = x.transpose(2, 1).contiguous()
        seq_len = x.size(1)
        pos_enc = self.get_sinusoidal_encoding(seq_len, self.d_model, x.device)
        x = x + pos_enc
        return x
    
    def get_sinusoidal_encoding(self, length, dim, device):
        pe = torch.zeros(length, dim, device=device)
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float, device=device) * 
            -(math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                        
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
if __name__ == "__main__":
    print("=" * 70)
    print("ORIGINAL MODEL TEST")
    print("=" * 70)
    
    batch_size = 4
    time_steps = 2000
    n_mels = 80
    
    input_tensor = torch.randn(batch_size, n_mels, time_steps)
    model = AudioEncoderEmbedding()
    output = model(input_tensor)
    
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (batch={batch_size}, time={time_steps//2}, d_model=256)")