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
                y (torch tensor): [Batchsize, seqlen] (Dữ liệu đầu vào đã token hóa)
                hidden (tuple(h, c)): Trạng thái ẩn từ bước trước đó (tùy chọn)
            Returns:
                gu: [Batchsize, seqlen, hidden_dim] (Output của Predictor)
                (h, c): Trạng thái ẩn mới
            """
            # 1. Chuyển đổi tokens thành embeddings
            # y_embedd shape: [Batchsize, seqlen, embedding_dim]
            y_embedd = self.embedding_layer(y)

            # 2. Đưa qua lớp LSTM
            # lstm_out shape: [Batchsize, seqlen, hidden_dim]
            # hidden shape: (h_n, c_n) mỗi cái là [num_layer, Batchsize, hidden_dim]
            lstm_out, (h, c) = self.lstm(y_embedd, hidden)
        
            # 3. Đưa qua lớp Linear và Dropout để tinh chỉnh feature
            # gu shape: [Batchsize, seqlen, hidden_dim]
            gu = self.linear(lstm_out)
            gu = self.dropout(gu)

            return gu, (h, c)
        
if __name__=="__main__":
    predictor = Predictor(vocab_size=10000, embedding_dim=768, hidden_dim=512, num_layer=5, dropout=0.1, maxLen=1024)
    inputs = torch.randint(0, 10000, (2, 256))
    outputs = predictor(inputs)
    print(outputs[0].shape)