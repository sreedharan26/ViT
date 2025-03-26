import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_size = emb_size
        
        super().__init__()
        
        self.projection = nn.Sequential(
            # Break the image into patches and then flatten them
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.embedding_size, 
                      kernel_size=self.patch_size, 
                      stride=self.patch_size),
            nn.Flatten(2, 3),
        )
        
    def forward(self, x):
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x
    

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 12, dropout: float = 0.1):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=self.emb_size, 
                                               num_heads=self.num_heads, 
                                               dropout=dropout,
                                               batch_first=True)
        
        self.norm = nn.LayerNorm(self.emb_size)
        
    def forward(self, x):
        x = self.norm(x)
        x, _ = self.attention(query=x, key=x, value=x, need_weights=False)
        return x
    

class MLPBlock(nn.Module):
    def __init__(self, in_features: int = 768, hidden_features: int = 3072, dropout: float = 0.1):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dropout = dropout
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.in_features),
            nn.Linear(in_features=self.in_features, out_features=self.hidden_features),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.hidden_features, out_features=self.in_features),
            nn.Dropout(p=self.dropout)
        )
    
    def forward(self, x):
        return self.mlp(x)
    

class TransformerEncodingBlock(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 12, mlp_hidden_features: int = 3072, dropout: float = 0.1):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.mlp_hidden_features = mlp_hidden_features
        self.dropout = dropout
        
        super().__init__()
        
        self.attention_block = MultiheadSelfAttentionBlock(emb_size=self.emb_size, num_heads=self.num_heads, dropout=self.dropout)
        self.mlp_block = MLPBlock(in_features=self.emb_size, hidden_features=self.mlp_hidden_features, dropout=self.dropout)
        
    def forward(self, x):
        x = self.attention_block(x) + x
        x = self.mlp_block(x) + x
        return x