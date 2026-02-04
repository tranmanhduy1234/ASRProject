import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sinusoidal_embeddings(n_pos, d_model):
    """Tạo bảng mã hóa vị trí Sinusoidal chuẩn"""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    print(inv_freq.shape)
    pos_seq = torch.arange(n_pos - 1, -1, -1).float() 
    print(pos_seq.shape)
    sinusoid_inp = torch.ger(pos_seq, inv_freq)
    print(sinusoid_inp.shape)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    print(pos_emb.shape)
    return pos_emb.unsqueeze(0)

class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
        
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def _relative_shift(self, x):
        batch, head, time1, time2 = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(batch, head, time2 + 1, time1)
        return x[:, :, 1:, :].view_as(x[:, :, :time1, :time2])

    def forward(self, query, key, value, pos_emb, mask=None):
        batch = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        
        q = self.linear_q(query).view(batch, q_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)
        
        p = self.linear_pos(pos_emb).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)

        content_score = torch.matmul(q + self.pos_bias_u.unsqueeze(1), k.transpose(-2, -1))
        
        pos_score = torch.matmul(q + self.pos_bias_v.unsqueeze(1), p.transpose(-2, -1))
        
        pos_score = self._relative_shift(pos_score)

        scores = (content_score + pos_score) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # 5. Nhân với Value và Output
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.fc_out(context)
    
# --- Cấu hình tham số ---
d_model = 512
n_head = 8
batch_size = 2
seq_len = 100

mha_rel = RelativePositionalMultiHeadAttention(d_model, n_head)

# Giả lập input
x = torch.randn(batch_size, seq_len, d_model)

# Tạo Positional Embeddings [1, 100, 512]
pos_emb = get_sinusoidal_embeddings(seq_len, d_model)

# SỬA LỖI TẠI ĐÂY: Mở rộng pos_emb để khớp với batch_size của x
# Từ [1, 100, 512] -> [2, 100, 512]
pos_emb_expanded = pos_emb.expand(batch_size, -1, -1)

# Chạy Forward
output = mha_rel(query=x, key=x, value=x, pos_emb=pos_emb_expanded)

print(f"Thành công! Output shape: {output.shape}")