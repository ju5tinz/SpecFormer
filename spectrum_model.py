import torch
import torch.nn as nn
import torch.nn.functional as F
from talking_head_attention import TalkingHeadAttention

class SpectrumModel(nn.Module):
    def __init__(self, token_size, dict_size, embed_dim=256, num_heads=64, seq_max_len=40, penultimate_dim=512):
        super().__init__()
        self.pre_attn_proj = nn.Linear(token_size, embed_dim)

        self.global_data_projector = nn.Linear(2, embed_dim)
        self.global_data_layernorm = nn.LayerNorm(embed_dim)

        self.pos_embedding = nn.Embedding(seq_max_len, embed_dim)

        self.pre_attn_layernorm = nn.LayerNorm(embed_dim)
        
        self.attention = TalkingHeadAttention(embed_dim, num_heads=num_heads, batch_first=True)

        self.post_attn_layernorm1 = nn.LayerNorm(embed_dim)

        self.post_attn_FFN1 = nn.Linear(embed_dim, embed_dim)

        self.post_attn_FFN2 = nn.Linear(embed_dim, embed_dim)

        self.post_attn_layernorm2 = nn.LayerNorm(embed_dim)

        self.penultimate_proj = nn.Linear(embed_dim, penultimate_dim)
        self.penultimate_norm = nn.LayerNorm(penultimate_dim)

        self.output_proj = nn.Linear(penultimate_dim, dict_size)
        

    def forward(self, x, charges, NCEs):
        # x shape: (batch_size, src_len, token_size)
        batch_size, src_len, token_size = x.shape
        device = x.device  # Ensure all new tensors are on the same device as input
        #print(f"Input shape: {x.shape}")
        x = self.pre_attn_proj(x)

        global_data = torch.stack([charges, NCEs], dim=1).float().to(device)
        global_embed = self.global_data_projector(global_data)
        global_embed = self.global_data_layernorm(global_embed)
        x = x + global_embed.unsqueeze(1) # global_embed is broadcasted to (batch_size, src_len, embed_dim)

        pos_src = torch.arange(0, src_len, device=device).unsqueeze(1).expand(src_len, batch_size)
        x = x + self.pos_embedding(pos_src).transpose(0, 1)

        pre_attn_x_norm = self.pre_attn_layernorm(x)
        #print(f"After projection and layer norm shape: {x.shape}")

        attn_output, _ = self.attention(pre_attn_x_norm, pre_attn_x_norm, pre_attn_x_norm)
        #print(f"After attention shape: {x.shape}")

        x = pre_attn_x_norm + attn_output

        post_attn_x_norm = self.post_attn_layernorm1(x)

        x = self.post_attn_FFN1(post_attn_x_norm)
        x = F.relu(x)

        x = self.post_attn_FFN2(x)

        x = post_attn_x_norm + x
        x = self.post_attn_layernorm2(x)

        x = self.penultimate_proj(x)
        x = self.penultimate_norm(x)
        x = F.relu(x)

        x = self.output_proj(x)
        #x = F.sigmoid(x)
        x = x.mean(dim=1)
        
        return x