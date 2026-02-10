import torch
import torch.nn as nn
import math

class AudioEncoderEmbedding(nn.Module):
    def __init__(self, n_mels=80, d_model=512, bias=False):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, bias=bias)
        self.norm1 = nn.GroupNorm(8, d_model)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=bias)
        self.norm2 = nn.GroupNorm(8, d_model)
        self.gelu2 = nn.GELU()
        
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1, bias=bias)
        self.norm3 = nn.GroupNorm(8, d_model)
        self.gelu3 = nn.GELU()
        
        self.d_model = d_model
        self.reset_parameters()
        
    def forward(self, x):
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        
        x = x
        return x
    
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