import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads=8, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (query_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)  # Project key to query_dim
        self.v_proj = nn.Linear(key_value_dim, query_dim)  # Project value to query_dim
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask=None):
        B, N_q, C_q = query.shape
        B, N_kv, C_kv = key.shape

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :], -torch.inf)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N_q, C_q)
        out = self.out_proj(out)
        return out

        # Feature Fusion
        self.cross_attention = CrossAttention(query_dim=vit_output_dim, key_value_dim=ego_mlp_output_dim, num_heads=num_attention_heads)
        self.fusion_norm = nn.LayerNorm(vit_output_dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(vit_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vit_output_dim),
        )

        # Add a sequence dimension of 1 to embeddings for attention
        status_embedding_seq = status_embedding.unsqueeze(1) # (B, 1, hidden_dim)
        vit_embedding_seq = vit_embedding.unsqueeze(1)     # (B, 1, vit_output_dim)

        # Cross-Attention: Visual attends to Ego-State (or vice-versa)
        attended_visual = self.cross_attention(query=vit_embedding_seq, key=status_embedding_seq, value=status_embedding_seq)
        fused_feature = self.fusion_norm(vit_embedding + attended_visual.squeeze(1)) # Residual connection and normalization
        fused_feature = self.fusion_ffn(fused_feature)