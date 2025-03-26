import torch
from torch import nn

from layers import PatchEmbedding, TransformerEncodingBlock

class VisionTransformer(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, num_classes: int = 3, 
                 emb_size: int = 768, depth: int = 12, num_heads: int = 12, mlp_dim: int = 3072, in_channels: int = 3,
                 dropout: float = 0.1):
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        
        super().__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.no_of_patches = (image_size // patch_size) ** 2
        
        self.class_token_embedding = nn.Parameter(torch.randn(1, 1, self.emb_size), requires_grad=True)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.no_of_patches + 1, self.emb_size), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        self.patch_embedding = PatchEmbedding(in_channels=self.in_channels, 
                                              patch_size=self.patch_size, 
                                              emb_size=self.emb_size, 
                                              )
        
        self.encoding_blocks = nn.ModuleList([
            TransformerEncodingBlock(emb_size=self.emb_size, 
                                     num_heads=self.num_heads, 
                                     mlp_hidden_features=self.mlp_dim, 
                                     dropout=self.dropout)
            for _ in range(self.depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.emb_size),
            nn.Linear(in_features=self.emb_size, out_features=self.num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        
        class_token = self.class_token_embedding.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x += self.positional_embedding
        x = self.embedding_dropout(x)

        for encoding_block in self.encoding_blocks:
            x = encoding_block(x)
        
        x = x[:, 0]
        x = self.mlp_head(x)
        return x