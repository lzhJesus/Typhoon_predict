from torch import nn

# class TransformerModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
#         super(TransformerModel, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, 
#                                           num_encoder_layers=num_layers, 
#                                           num_decoder_layers=num_layers, 
#                                           dropout=dropout)
#         self.fc_out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, src, tgt):
#         src = self.embedding(src)  # [batch_size, seq_len, hidden_dim]
#         tgt = self.embedding(tgt)  # [batch_size, seq_len, hidden_dim]
        
#         # 转置形状以匹配 PyTorch 的 Transformer
#         src = src.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
#         tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        
#         output = self.transformer(src, tgt)
#         return self.fc_out(output)
    

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)  # out 的形状: (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out