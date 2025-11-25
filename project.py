import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import tiktoken

# 加载数据
with open(r'.\sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 参数设置 
batch_size = 4
context_length = 16
d_model = 64

# 创建字符映射
encoder = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoder.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long)
max_token_value = tokenized_text.max().item()

# data split
train_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_idx]
val_data = tokenized_text[train_idx:]

data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])

# word embedding
input_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)

# position encoding
position_encoding_lookup_table = nn.Embedding(context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
div_terms = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
with torch.no_grad():
    position_encoding_lookup_table.weight[:, 0::2] = torch.sin(position * div_terms)
    position_encoding_lookup_table.weight[:, 1::2] = torch.cos(position * div_terms)

position_encoding = position_encoding_lookup_table.weight.unsqueeze(0).expand(batch_size, -1, -1)
x = x_batch_embedding + position_encoding
y = y_batch_embedding + position_encoding

# QKV
Wq, Wk, Wv = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
Q, K, V = Wq(x), Wk(x), Wv(x)

# Multi-Head Attention
num_heads = 4
Q = Q.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
K = K.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
V = V.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)

attn = Q @ K.transpose(-2, -1) / math.sqrt(d_model // num_heads)

# mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
mask = mask.unsqueeze(0).unsqueeze(0)
attn = attn.masked_fill(mask, float('-inf'))

# softmax
attention_score = F.softmax(attn, dim=-1)
A = attention_score @ V

#  heads
A = A.transpose(1, 2).reshape(batch_size, -1, d_model)
Wo = nn.Linear(d_model, d_model)
attn_out = Wo(A)

#  LayerNorm
layer_norm = nn.LayerNorm(d_model)
out1 = layer_norm(attn_out + x)

# feed-forward network
ffn = nn.Sequential(
    nn.Linear(d_model, d_model * 4),
    nn.ReLU(),
    nn.Linear(d_model * 4, d_model)
)
out2 = ffn(out1)
out2 = layer_norm(out2 + out1)

# output
output_layer = nn.Linear(d_model, max_token_value + 1)
logits = F.softmax(output_layer(out2), dim=-1)
prediction = torch.argmax(logits[0, 0], dim=-1).item()
print(encoder.decode([prediction]))
