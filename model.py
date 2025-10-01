import math
import torch
from torch import nn

class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
    ):
        seq_len_q, batch_size, _ = query.shape 
        seq_len_kv = key.shape[0]

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=0)
            b_q, b_k, b_v = self.in_proj.bias.chunk(3, dim=0)
            q = nn.functional.linear(query, w_q, b_q)
            k = nn.functional.linear(key, w_k, b_k)
            v = nn.functional.linear(value, w_v, b_v)

        q = q.view(seq_len_q, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )
        k = k.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )
        v = v.view(seq_len_kv, batch_size, self.num_heads, self.head_dim).permute(
            1, 2, 0, 3
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        context = (
            context.permute(2, 0, 1, 3)
            .contiguous()
            .view(seq_len_q, batch_size, self.embed_dim)
        )

        attn_output = self.out_proj(context)

        if need_weights:
            return attn_output, attn_weights.mean(dim=1)
        else:
            return attn_output, None