import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=5000):
        super(TransformerModel, self).__init__()

        # Embedding Layers
        self.src_embedding = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        # Transformer Model
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )

        # Extra Layer 1: Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Extra Layer 2: Additional Feedforward Layer
        self.extra_fc = nn.Linear(embed_dim, embed_dim)

        # Output Layer
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))  # Shape: (batch, seq, embed_dim)
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))  

        # **Transpose for Transformer Input**: (seq, batch, embed_dim)
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)  # (seq, batch, embed_dim)

        # Apply Extra Layers
        output = self.layer_norm(output)  # Normalize output
        output = F.relu(self.extra_fc(output))  # Feedforward layer with ReLU activation

        output = self.fc_out(output.permute(1, 0, 2))  # Back to (batch, seq, vocab_size)
        
        return output
