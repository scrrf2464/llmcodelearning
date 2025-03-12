import math
import torch
import torch.nn as nn
from torchsummary import summary
class MultiHeadSelfattention(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num #head_dim做selfattention
        # self.attention_mask = attention_mask
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, X, attention_mask=None):
        batch, seq_len, h = X.size()
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        #(batch, seq_len, h) => (batch, head_num, seq_len, gead_dim) (h=> head_num*head_dim)
        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1,2)

        attention_weight = torch.matmul(q_state,k_state.transpose(-1,-2))/math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0, float('-inf')
            )

        attention_weight = torch.softmax(attention_weight, -1)
        attention_weight = self.dropout(attention_weight)

        output_mid = torch.matmul(attention_weight, v_state)

        output_mid = output_mid.transpose(1,2).contiguous()
        output_mid = output_mid.view(batch, seq_len, -1)

        output = self.output_proj(output_mid)

        return output

if __name__ == '__main__':
    attention_mask = (
        torch.tensor(
            [
                [0, 1],
                [0, 0],
                [1, 0],
            ]
        )
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(3, 8, 2, 2)
    )

    x = torch.rand(3, 2, 128)
    net = MultiHeadSelfattention(128, 8)
    # print(net(x, attention_mask).shape)
    # print(net)
    summary(net, input_size=(2, 128), batch_size=5)