import torch
from torch import nn
import math

class LearningPositionEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, inputs_embeds):

        bsz, seq_len = inputs_embeds.shape[:2]
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1).to(inputs_embeds.device)
        positions_embeddings = super().forward(positions + self.offset)

        return inputs_embeds + positions_embeddings

class StaticPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):

        super(StaticPositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, inputs_embeds):
        seq_len = inputs_embeds.size(1)
        return inputs_embeds + self.pe[:, :seq_len, :]

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        self.dropout =dropout

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training) 
        x = self.fc2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

class CoordinateMapping(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CoordinateMapping, self).__init__()
        
        self.mapping_x = nn.Linear(in_feat, out_feat)
        self.mapping_y = nn.Linear(in_feat, out_feat)

    def forward(self, x_coord, y_coord):
        
        x_embed = self.mapping_x(x_coord)       

        y_embed = self.mapping_y(y_coord)        
        
        return x_embed, y_embed
    

if __name__ == '__main__':
    c = CoordinateMapping(256, 512)
    x = torch.randn(2, 180, 2, 256)
    # y = torch.randn(2, 180, 256)
    
    x_embed, y_embed = c(x[..., 0], x[..., 1])
    print(x_embed.shape, y_embed.shape)