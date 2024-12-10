import torch
import torch.nn as nn
import math

class LlamaConfig:
    def __init__(self, vocab_size, dim, n_layers, n_heads):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = 0.1

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.scale

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.o_proj = nn.Linear(config.dim, config.dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for multi-head attention
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match multi-head attention shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

        # Softmax and dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        context = (attn_probs @ v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.dim)

        return self.o_proj(context)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.dim * 4)
        self.w2 = nn.Linear(config.dim * 4, config.dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, x, attention_mask=None):
        # Residual connection with pre-norm
        h = x + self.attention(self.attention_norm(x), attention_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(2048, config.dim)  # Assuming max sequence length
        
        self.blocks = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim)
        self.output_layer = nn.Linear(config.dim, config.vocab_size)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_emb(x)
        
        # Positional embeddings
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Apply attention mask to transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final normalization and output projection
        x = self.norm(x)
        return self.output_layer(x)