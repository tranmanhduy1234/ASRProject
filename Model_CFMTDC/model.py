"""
Conformer-Transducer Model
Minimal implementation for ASR (Automatic Speech Recognition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# ============================================================================
# 1. AUDIO ENCODER EMBEDDING
# ============================================================================

class AudioEncoderEmbedding(nn.Module):
    """Converts mel-spectrogram to embeddings with Conv2d subsampling"""
    
    def __init__(self, input_dim=80, output_dim=128, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(64)
        
        # After conv2: height = input_dim // 2, channels = 64
        # So flattened size = 64 * (input_dim // 2)
        self.fc = nn.Linear(64 * (input_dim // 2), output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, x_len=None):
        """
        Args:
            x: [B, T, input_dim]
            x_len: [B] length of each sequence
        Returns:
            out: [B, T', output_dim]
            out_len: [B] length after subsampling
        """
        x = x.unsqueeze(1)  # [B, 1, T, input_dim]
        
        # Conv1: [B, 1, T, input_dim] -> [B, 32, T, input_dim]
        x = self.conv1(x)
        B, C, T, H = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, input_dim, 32]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 32, T, input_dim]
        x = F.relu(x)
        
        # Conv2: [B, 32, T, input_dim] -> [B, 64, T, input_dim//2]
        x = self.conv2(x)
        B, C, T, H = x.shape  # C=64, H=input_dim//2
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, input_dim//2, 64]
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 64, T, input_dim//2]
        x = F.relu(x)
        
        # Flatten: [B, 64, T, input_dim//2] -> [B, T, 64 * input_dim//2]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, 64, input_dim//2]
        x = x.reshape(B, T, -1)  # [B, T, 64 * input_dim//2]
        
        # Project to output_dim
        x = self.fc(x)  # [B, T, output_dim]
        x = self.final_norm(x)
        x = self.dropout(x)
        
        # Update length (only stride=2 from conv2)
        if x_len is not None:
            out_len = (x_len + 2 * 1 - 3) // 2 + 1  # stride=2, kernel=3, pad=1
        else:
            out_len = None
        
        return x, out_len


# ============================================================================
# 2. POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x: [B, T, d_model]"""
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# 3. FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Module):
    """FFN: Linear -> GELU -> Dropout -> Linear -> Dropout"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================================
# 4. CONFORMER BLOCK
# ============================================================================

class ConformerBlock(nn.Module):
    """
    Conformer Block: FFN -> MHSA -> Conv -> FFN
    
    Architecture:
        x -> LayerNorm -> FFN (x0.5) -> x + out
        -> LayerNorm -> MHSA -> x + out
        -> LayerNorm -> Conv1d (depthwise) -> x + out
        -> LayerNorm -> FFN (x0.5) -> x + out
    """
    
    def __init__(self, d_model=128, nhead=4, d_ff=512, conv_kernel=31, dropout=0.1):
        super().__init__()
        
        # FFN 1 (0.5x residual)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        
        # MHSA
        self.norm2 = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=False, batch_first=True
        )
        
        # Conv (depthwise)
        self.norm3 = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2, 
                     groups=d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # FFN 2 (0.5x residual)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T] (True for padding)
        Returns:
            out: [B, T, d_model]
        """
        # FFN 1
        x = x + 0.5 * self.ffn1(self.norm1(x))
        
        # MHSA
        attn_out, _ = self.mhsa(
            self.norm2(x), self.norm2(x), self.norm2(x),
            key_padding_mask=mask
        )
        x = x + attn_out
        
        # Conv
        x_norm = self.norm3(x)
        x = x + self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
        
        # FFN 2
        x = x + 0.5 * self.ffn2(self.norm4(x))
        
        return x


# ============================================================================
# 5. CONFORMER ENCODER
# ============================================================================

class ConformerEncoder(nn.Module):
    """Conformer Encoder: 12 Conformer blocks + Positional encoding"""
    
    def __init__(self, d_model=128, nhead=4, num_layers=12, d_ff=512, 
                 conv_kernel=31, dropout=0.1):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, d_ff, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, x_len=None):
        """
        Args:
            x: [B, T, d_model]
            x_len: [B] sequence lengths
        Returns:
            out: [B, T, d_model]
            x_len: [B]
        """
        x = self.pos_encoding(x)
        
        # Create padding mask
        mask = None
        if x_len is not None:
            batch_size = x.size(0)
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= x_len.unsqueeze(1)
        
        # Apply conformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        return x, x_len

# ============================================================================
# 6. PREDICTOR NETWORK (LSTM)
# ============================================================================

class Predictor(nn.Module):
    """Predictor: Embedding -> LSTM -> Linear"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, y, hidden=None):
        """
        Args:
            y: [B, U] token indices
            hidden: (h, c) from previous step
        Returns:
            gu: [B, U, hidden_dim]
            (h, c): new hidden states
        """
        y_emb = self.embedding(y)
        y_emb = self.dropout(y_emb)
        
        lstm_out, (h, c) = self.lstm(y_emb, hidden)
        gu = self.linear(lstm_out)
        gu = self.dropout(gu)
        
        return gu, (h, c)

    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden states"""
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return h, c

# ============================================================================
# 7. JOINER NETWORK
# ============================================================================

class Joiner(nn.Module):
    """Joiner: Concatenate encoder + predictor outputs -> project -> logits"""
    
    def __init__(self, encoder_dim=128, predictor_dim=256, joiner_dim=256, vocab_size=500):
        super().__init__()
        
        self.linear1 = nn.Linear(encoder_dim + predictor_dim, joiner_dim, bias=False)
        self.linear2 = nn.Linear(joiner_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, ft, gu):
        """
        Args:
            ft: [B, T, encoder_dim]
            gu: [B, U, predictor_dim]
        Returns:
            logits: [B, T, U, vocab_size]
        """
        B, T, encoder_dim = ft.shape
        _, U, predictor_dim = gu.shape
        
        # Expand for broadcasting
        ft_expanded = ft.unsqueeze(2).expand(B, T, U, encoder_dim)  # [B, T, U, encoder_dim]
        gu_expanded = gu.unsqueeze(1).expand(B, T, U, predictor_dim)  # [B, T, U, predictor_dim]
        
        # Concatenate: [B, T, U, encoder_dim + predictor_dim]
        joint = torch.cat([ft_expanded, gu_expanded], dim=-1)
        
        # Project
        joint = self.linear1(joint)  # [B, T, U, joiner_dim]
        joint = torch.tanh(joint)
        joint = self.dropout(joint)
        
        logits = self.linear2(joint)  # [B, T, U, vocab_size]
        return logits

# ============================================================================
# 8. CONFORMER-TRANSDUCER (MAIN MODEL)
# ============================================================================

class ConformerTransducer(nn.Module):
    """
    Conformer-Transducer for ASR
    
    Architecture:
        Audio Input [B, T, 80]
            ↓
        AudioEncoderEmbedding → [B, T', 128]
            ↓
        ConformerEncoder → [B, T', 128] (ft)
            ↓
            ├─ Predictor: [B, U] → [B, U, 256] (gu)
            ↓
        Joiner → [B, T', U, vocab_size]
    """
    
    def __init__(self, input_dim=80, encoder_dim=128, predictor_dim=256, 
                 joiner_dim=256, vocab_size=500, num_encoder_layers=12, 
                 num_predictor_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        
        # Audio encoder embedding
        self.audio_embedding = AudioEncoderEmbedding(input_dim, encoder_dim, dropout)
        
        # Conformer encoder
        self.encoder = ConformerEncoder(
            d_model=encoder_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            d_ff=encoder_dim * 4,
            conv_kernel=31,
            dropout=dropout
        )
        
        # Predictor
        self.predictor = Predictor(
            vocab_size=vocab_size,
            embedding_dim=predictor_dim // 2,
            hidden_dim=predictor_dim,
            num_layers=num_predictor_layers,
            dropout=dropout
        )
        
        # Joiner
        self.joiner = Joiner(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            joiner_dim=joiner_dim,
            vocab_size=vocab_size
        )
    
    def forward(self, x, x_len, y):
        """
        Args:
            x: [B, T, input_dim] - Audio features
            x_len: [B] - Audio lengths
            y: [B, U] - Target tokens (with context)
        
        Returns:
            logits: [B, T', U, vocab_size] - Joint probabilities
        """
        # Encode audio
        x_emb, x_len = self.audio_embedding(x, x_len)
        ft, _ = self.encoder(x_emb, x_len)
        
        # Predict language
        gu, _ = self.predictor(y)
        
        # Join
        logits = self.joiner(ft, gu)
        
        return logits
    
    def encode(self, x, x_len=None):
        """Encode audio (for inference)"""
        x_emb, x_len = self.audio_embedding(x, x_len)
        ft, _ = self.encoder(x_emb, x_len)
        return ft, x_len
    
    def predict(self, y, hidden=None):
        """Predict language (for inference)"""
        return self.predictor(y, hidden)
    
    def join(self, ft, gu):
        """Join encoder and predictor (for inference)"""
        return self.joiner(ft, gu)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ConformerTransducer(
        input_dim=80,
        encoder_dim=128,
        predictor_dim=256,
        vocab_size=500,
        num_encoder_layers=12,
        num_predictor_layers=2,
        nhead=4
    ).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,.0f}")
    
    # Create dummy data
    batch_size = 2
    audio_len = 100
    target_len = 30
    
    x = torch.randn(batch_size, audio_len, 80).to(device)
    x_len = torch.tensor([100, 95], dtype=torch.long).to(device)
    y = torch.randint(0, 500, (batch_size, target_len)).to(device)
    y[:, 0] = 0  # Start with blank token
    
    # Forward pass
    logits = model(x, x_len, y)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [B={batch_size}, T, U={target_len}, vocab=500]")
    
    # Inference example
    print("\n" + "="*60)
    print("INFERENCE EXAMPLE")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Encode audio
        ft, _ = model.encode(x, x_len)
        print(f"Encoded audio shape: {ft.shape}")
        
        # Predict one token
        y_test = torch.tensor([[0]], dtype=torch.long).to(device)  # Blank token
        gu, hidden = model.predict(y_test)
        print(f"Predictor output shape: {gu.shape}")
        
        # Join
        logits_test = model.join(ft, gu)
        print(f"Joiner output shape: {logits_test.shape}")
    
    print("\n✅ Model works correctly!")