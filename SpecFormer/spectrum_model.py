import torch
import torch.nn as nn
import torch.nn.functional as F
from talking_head_attention import TalkingHeadAttention

class SpectrumModel(nn.Module):
    """
    Neural network model for spectrum prediction using talking-head attention.
    """
    def __init__(self, token_size: int, dict_size: int, *, embed_dim: int = 256, num_heads: int = 64, seq_max_len: int = 40, penultimate_dim: int = 2048, dropout_rate: float = 0.1):
        super().__init__()
        self.pre_attn_proj = nn.Linear(token_size, embed_dim)

        self.global_data_projector = nn.Linear(2, embed_dim)
        self.global_data_layernorm = nn.LayerNorm(embed_dim)

        self.pos_embedding = nn.Embedding(seq_max_len, embed_dim)

        self.pre_attn_layernorm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = TalkingHeadAttention(embed_dim, num_heads=num_heads, batch_first=True)

        self.post_attn_layernorm1 = nn.LayerNorm(embed_dim)

        self.post_attn_FFN1 = nn.Linear(embed_dim, embed_dim)
        self.post_attn_FFN2 = nn.Linear(embed_dim, embed_dim)

        self.post_attn_layernorm2 = nn.LayerNorm(embed_dim)

        self.penultimate_proj = nn.Linear(embed_dim, penultimate_dim)
        self.penultimate_norm = nn.LayerNorm(penultimate_dim)

        self.output_proj = nn.Linear(penultimate_dim, dict_size)

    def forward(self, x, charges, NCEs):
        """
        Forward pass for spectrum prediction.
        Args:
            x: (batch_size, src_len, token_size) input sequence tensor
            charges: (batch_size,) tensor
            NCEs: (batch_size,) tensor
        Returns:
            (batch_size, dict_size) output tensor
        """
        # x shape: (batch_size, src_len, token_size)
        batch_size, src_len, token_size = x.shape
        device = x.device  # Ensure all new tensors are on the same device as input

        x = self.pre_attn_proj(x)

        global_data = torch.stack([charges, NCEs], dim=1).float().to(device)
        global_embed = self.global_data_projector(global_data)
        global_embed = self.global_data_layernorm(global_embed)
        x = x + global_embed.unsqueeze(1) # global_embed is broadcasted to (batch_size, src_len, embed_dim)

        pos_src = torch.arange(0, src_len, device=device).unsqueeze(1).expand(src_len, batch_size)
        x = x + self.pos_embedding(pos_src).transpose(0, 1)

        # Apply dropout to the combined embeddings
        x = self.dropout(x)

        pre_attn_x_norm = self.pre_attn_layernorm(x)

        attn_output, _ = self.attention(pre_attn_x_norm, pre_attn_x_norm, pre_attn_x_norm)
        
        # Apply dropout after attention (residual connection)
        x = pre_attn_x_norm + self.dropout(attn_output)

        post_attn_x_norm = self.post_attn_layernorm1(x)

        # FFN block
        ffn_output = self.post_attn_FFN1(post_attn_x_norm)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.dropout(ffn_output)  # Dropout after activation
        
        ffn_output = self.post_attn_FFN2(ffn_output)

        # Apply dropout after the second FFN (residual connection)
        x = post_attn_x_norm + self.dropout(ffn_output)
        x = self.post_attn_layernorm2(x)

        # Penultimate block
        x = self.penultimate_proj(x)
        x = self.penultimate_norm(x)
        x = F.relu(x)
        x = self.dropout(x) # Dropout after final activation before output layer

        # Output block
        x = self.output_proj(x)
        x = torch.sigmoid(x)
        x = x.mean(dim=1)
        
        return x