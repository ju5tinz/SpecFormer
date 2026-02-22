import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class TalkingHeadAttention(nn.Module):
    """
    Simplified implementation of Talking-Head Attention.
    Assumes embed_dim is divisible by num_heads and no attention masking.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, batch_first: bool = False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Simplified projection layers, all mapping embed_dim to embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # The core "talking-head" layers
        self.talk_pre_softmax = nn.Linear(num_heads, num_heads, bias=False)
        self.talk_post_softmax = nn.Linear(num_heads, num_heads, bias=False)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for talking-head attention.
        Args:
            query, key, value: (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) if batch_first
        Returns:
            output: attended tensor
            attn_weights: optional attention weights
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        # 1. Project and reshape Q, K, V for multi-head processing
        q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        
        # Resulting shapes are (bsz, num_heads, seq_len, head_dim)

        # 2. Calculate scaled dot-product attention scores
        q = q / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # 3. "Talk" before softmax by applying a linear layer across heads
        attn_scores = attn_scores.permute(0, 2, 3, 1) # -> (N, L, S, H)
        attn_scores = self.talk_pre_softmax(attn_scores)
        attn_scores = attn_scores.permute(0, 3, 1, 2) # -> (N, H, L, S)

        # 4. Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 5. "Talk" after softmax
        attn_weights = attn_weights.permute(0, 2, 3, 1) # -> (N, L, S, H)
        attn_weights = self.talk_post_softmax(attn_weights)
        attn_weights = attn_weights.permute(0, 3, 1, 2) # -> (N, H, L, S)

        # 6. Compute context vector and project the output
        context = torch.matmul(attn_weights, v)
        context = context.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        output = self.out_proj(context)

        if self.batch_first:
            output = output.transpose(1, 0)

        # 7. Optionally return attention weights
        if need_weights:
            final_weights = attn_weights.mean(dim=1) if average_attn_weights else attn_weights
            return output, final_weights
        
        return output, None