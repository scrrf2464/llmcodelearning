import torch
import torch.nn as nn
import math

class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim: int = 728) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)

    def forward(self, X):
        # x:(batchsize, seq_len, hidden_dim)
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        #attention_value:(batchsize, seq_len, seq_len)
        attention_value = torch.matmul(Q, K.transpose(-1,-2))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1 )
        #开根号的原因是防止成绩梯度消失。 softmax 会让一个值很大，其他值很小，原本较小的值就会接近于0.丢失梯度

        print(attention_weight)

        output = torch.matmul(attention_weight, V)
        return output

class SelfAttentionV2(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        #若网络较小，可以将qkv合为一个较大的张量，统一运算，效率优化
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim*3)

    def forward(self, X):
        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)

        attention_value = torch.matmul(Q, K.transpose(-1,-2))
        attention_weight = torch.softmax(attention_value / math.sqrt(self.hidden_dim), dim=-1)

        # output = torch.matmul(attention_weight, V)
        output = attention_weight @ V #@也可表示为矩阵乘法
        return output

"""
torch.Tensor.masked_fill()
masked_fill() 是一个非常有用的方法，它允许你根据一个条件（通常是一个布尔掩码）来填充（替换）张量中的值。

参数
mask (Tensor): 一个与原张量形状相同的布尔类型张量（即包含 True 或 False 值）。True 表示需要替换该位置的元素。
value (Scalar): 要填充的值。可以是任何标量类型，如整数、浮点数等，取决于张量的数据类型。
"""
class SelfAttentionV3(nn.Module):
    def __init(self, hidden_dim: int, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask = None):
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)

        attention_value = torch.matmul(Q, K.transpose(-1,-2))/ math.sqrt(self.hidden_dim)

        if attention_mask is not None:
            attention_value = attention_value.masked_fill(attention_mask, '1e-20')
        attention_weight = torch.softmax(attention_value, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)

        output = torch.matmul(attention_weight, V)
        output = self.output_proj(output)
        return output

# x = torch.rand(3,2,4)
# model = SelfAttentionV1(4)
# model.forward(x)