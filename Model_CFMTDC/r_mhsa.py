import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sinusoidal_embeddings(n_pos, d_model):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pos_seq = torch.arange(n_pos - 1, -1, -1).float() 
    sinusoid_inp = torch.ger(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)

class RelativePositionalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.bias = bias
        
        self.linear_q = nn.Linear(d_model, d_model, self.bias)
        self.linear_k = nn.Linear(d_model, d_model, self.bias)
        self.linear_v = nn.Linear(d_model, d_model, self.bias)
        
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.linear_pos = nn.Linear(d_model, d_model, bias=self.bias)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model, self.bias)
        
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def _relative_shift(self, x):
        batch, head, time1, time2 = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(batch, head, time2 + 1, time1)
        return x[:, :, 1:, :].view_as(x[:, :, :time1, :time2])

    def forward(self, query, key, value, mask=None):
        # mask [batch_size, seq_len]: 1: token, 0: padding
        
        batch = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        d_model = key.size(2)
        assert q_len == k_len, "Relative self-attention requires q_len == k_len"
        
        pos_emb = get_sinusoidal_embeddings(k_len, d_model).to(query.device)
        pos_emb = pos_emb.expand(batch, -1, -1)

        q = self.linear_q(query).view(batch, q_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)
        
        # pos_emb [2, 100, 512]
        p = self.linear_pos(pos_emb).view(batch, k_len, self.n_head, self.d_k).transpose(1, 2)
        
        content_score = torch.matmul(q + self.pos_bias_u.unsqueeze(1), k.transpose(-2, -1))
        
        pos_score = torch.matmul(q + self.pos_bias_v.unsqueeze(1), p.transpose(-2, -1))
        
        # pos_score: torch.Size([2, 8, 100, 100])
        pos_score = self._relative_shift(pos_score)
        
        scores = (content_score + pos_score) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_mask = mask[:, None, :, None] & mask[:, None, None, :]
            scores = scores.masked_fill(attn_mask == False, -1e9)
        print(attn_mask)
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # 5. Nhân với Value và Output
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, q_len, self.d_model)
        return self.fc_out(context)
    
if __name__=="__main__":
    mha_rel = RelativePositionalMultiHeadSelfAttention(512, 8)
    # Giả lập input
    x = torch.randn(1, 5, 512)
    mask = torch.ones(1, 5, dtype=torch.bool)
    mask[:, 2:] = False
    output = mha_rel(query=x, key=x, value=x, mask=mask)
    print(f"Thành công! Output shape: {output.shape}")