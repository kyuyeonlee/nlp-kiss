import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class KoreanSpacingTransformer(nn.Module):
    """
    Transformer-based sequence labeling model for Korean text spacing.
    Given input character indices, predicts binary labels (space/no-space) at each position.
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 num_labels: int = 2):
        super(KoreanSpacingTransformer, self).__init__()
        # Embedding layer (padding_idx=0 for <PAD>)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Final classifier: map d_model -> num_labels for each position
        self.classifier = nn.Linear(d_model, num_labels)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: LongTensor of shape (batch_size, seq_len) containing token indices.
            src_key_padding_mask: BoolTensor of shape (batch_size, seq_len), True for padding positions.
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, num_labels)
        """
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)

        # Transformer expects shape (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        # Pass through the encoder
        memory = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (seq_len, batch, d_model)
        # Back to (batch, seq_len, d_model)
        output = memory.transpose(0, 1)

        # Classifier per token
        logits = self.classifier(output)  # (batch, seq_len, num_labels)
        return logits

# Example usage:
# model = KoreanSpacingTransformer(vocab_size=len(char2idx))
# logits = model(input_batch, src_key_padding_mask=mask)  # logits.shape == (batch, seq_len, 2)
