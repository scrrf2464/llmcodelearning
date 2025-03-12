import math
import torch
import torch.nn as nn

class MultiHeadSelfattention(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout=0.1, attention_mask=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num #head_dim做selfattention
        self.attention_mask = attention_mask
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, X):
        batch, seq_len, h = X.size()
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        #(batch, seq_len, h) => (batch, head_num, seq_len, gead_dim) (h=> head_num*head_dim)
        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)

        attention_weight = torch.matmul(q_state,k_state.transpose(-1,-2))/math.sqrt(self.head_dim)

        if self.attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                self.attention_mask == 0, float('-inf')
            )

        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout(attention_weight)

        output_mid = torch.matmul(attention_weight, v_state)

        output_mid = output_mid.transpose(1,2).comtiguous()
        output_mid = output_mid.view(batch, seq_len, -1)

        output = self.output_proj(output_mid)

        return output


