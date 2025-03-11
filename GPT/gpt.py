# part 1: 导入相关的 package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import math

@dataclass
class GPTConfig:
    block_size: int = 512   # 这里其实应该是文本的最大长度（ max_seq_len）
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里我同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    # # tiktoken 使用的是 GPT-2 的词表，大约有 50257 个token
    vocab_size: int = 50257

class SingalHeadAttention(nn.Module):
    def __int__(self, config):
        super().__init__()

        self.key = nn.Linear(config.n_embd, config.head_size)
        self.querry = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size
        # self.attention_dropout = nn.Dropout(config.dropout)
        # self.output = nn.Linear(config.n_embd, config.n_embd)

        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        batch_size, seq_len, hidden_size = X.size()
        Q = self.querry(X)
        K = self.key(X)
        V = self.value(X)

        weight = torch.matmul(Q, K.transpose(-1,-2))

        # if attention_mask is not None:
        weight = weight.masked_fill(self.attention_mask[:seq_len, :seq_len] == 0,
                                    float('inf')
                                    )/ math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=1)
        weight = self.dropout(weight)
        output = torch.matmul(weight, V)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
            SingalHeadAttention(config)
            for _ in range(config.head) #创建config.head个单头注意力
        ]
        )
        self.proj = nn.Linear(config.n_embd,config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        #torch.cat() 是 PyTorch 中用于在指定维度上拼接（concatenate）张量的一个函数
        output = torch.cat(
            [h(X) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        #nn.Sequential 是 PyTorch 中的一个容器类，用于按顺序组合多个神经网络层（或模块）。
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 线性变换：输入维度 -> 4倍输入维度
            nn.GELU(),  # 非线性激活函数 GELU
            nn.Linear(4 * config.n_embd, config.n_embd),  # 线性变换：4倍输入维度 -> 输入维度
            nn.Dropout(config.dropout)  # Dropout 层，防止过拟合
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)  # 多头自注意力机制
        self.ffn = FeedForward(config)         # 前馈神经网络
        self.ln1 = nn.LayerNorm(config.n_embd) # 第一层归一化
        self.ln2 = nn.LayerNorm(config.n_embd) # 第二层归一化

    def forward(self, x):
        # 残差连接 + 第一层归一化 + 多头自注意力机制
        x = x + self.att(self.ln1(x))
        # 残差连接 + 第二层归一化 + 前馈神经网络
        x = x + self.ffn(self.ln2(x))
        return x

"""
输入张量
X: [batch_size, seq_len, n_embd]
1. 第一层归一化 + 多头自注意力机制 ： self.att(self.ln1(x))
1.1 层归一化（self.ln1()）
    input: [batch_size, seq_len, n_embd]    output:[batch_size, seq_len, n_embd]
1.2 多头注意力(self.att())
    多头注意力由n_head个单头组成，单头的输入输出如下：
    x: [batch_size, seq_len, n_embd]
    
    x的Q,K,V 线性变换,降低最后一维, head_size = n_embd // n_head
    Q = K = V = [batch_size, seq_len, head_size] 
    
    注意力权重为 Q*KT, [seq_len, head_size] * [head_size, seq_len]  
    weight -> [batch_size, seq_len, seq_len]
    weight.masked_fill, F.softmax 和 dropout 不影响维度
    
    加权求和输出值为 weight * V [seq_len, seq_len] * [seq_len, head_size] 
    output -> [batch_size, seq_len, head_size] 
    
    在最后一维拼接单头注意力输出 torch.cat(）
    input: [batch_size, seq_len, head_size] * n_head  
    output: [batch_size, seq_len, n_embd]
    
    线性投影， dropout维度不变(self.proj()，self.dropout())
    output: [batch_size, seq_len, n_embd]
    
1.3 残差连接 x = x + self.att()
    加法维度不变
    
2. 第二层归一化 + 前馈神经网络： x = x + self.ffn(self.ln2(x))
"""

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        #(embedding, position, norm, mlp, block)
        #Token embedding是指将词汇表中的每个单词或标记（token）映射到一个高维向量空间的过程。
        # Position embedding则是为了给模型提供关于句子中单词顺序的信息。
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #现在的模型会用tie weight 减小参数，
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 是输入的 token ids， taget是目标token ids 两者shape相同
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx) #(batchsize, seq_len, n_embd)

        # seq 长度是这次输入的最大长度
        pos_emb = self.position_embedding_table(
            # 要确保 位置编码和输入的 idx 在同一个设备上
            torch.arange(seq_len, device=idx.device)
        )

        # 有一个经典题目：为什么 embedding 和 position 可以相加？
        #在高维空间中，两个随机选择的向量几乎正交.这使得Transformer能相对独立地处理token和position的信息
        x = token_emb + pos_emb  # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # shape is (batch, seq_len, vocab_size)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        import tiktoken
        #这是gpt官方的tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        #"<|endoftext|>" [50256]
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json
        self.encoded_data = []
        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines: #防止数据过大爆内存，每次取1000行
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text) #List
            full_encoded.extend(encoded_text + [self.eos_token]) #将所有文本合为一行

        # 将长文本分割成训练样本512
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i + self.block_size + 1] #每一行有513个token
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

        def __len__(self):
            return len(self.encoded_data)

        def __getitem__(self, idx):
            chunk = self.encoded_data[idx] # #每一行有513个token
            x = torch.tensor(chunk[:-1], dtype=torch.long) #输入取前512个token, 提前完成了shift
            y = torch.tensor(chunk[1:], dtype=torch.long) #输出取后512个token
            return x, y

        def encode(self, text):
            """将文本编码为token IDs"""
            return self.enc.encode(text)

        def decode(self, ids):
            """将token IDs解码为文本"""
            return self.enc.decode(ids)

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits, loss = model(x, targets=y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 调整学习率
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss


def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

if __name__ == "main":
    train_dataset = MyDataset('/root/fs/mobvoi_seq_monkey_general_open_corpus.jsonl')
    # split traindataset to train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)
    model = GPT(GPTConfig())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 打印模型一共有多少参数

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # 设置 cosine 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    for epoch in range(2):
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
        val_loss = eval(model, val_loader, device)
        print(
            f'Epoch: {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

        # 保存模型
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        # 保存每个epoch的模型
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')