import torch
import torch.nn as nn
import torch.nn.functional as F

class KoreanSpacingCNN(nn.Module):
    """
    CNN-based sequence labeling model for Korean text spacing.
    Given input character indices, predicts binary labels (space/no-space) at each position.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 num_filters: int = 200,
                 kernel_sizes: list = [3, 5, 7],
                 dropout: float = 0.3,
                 num_labels: int = 2):
        super(KoreanSpacingCNN, self).__init__()
        # Embedding layer (padding_idx=0 for <PAD>)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Convolutional layers with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # "same" padding to preserve sequence length
            )
            for k in kernel_sizes
        ])

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Final projection: from concatenated conv features to label logits per position
        self.classifier = nn.Conv1d(
            in_channels=num_filters * len(kernel_sizes),
            out_channels=num_labels,
            kernel_size=1
        )

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (batch_size, seq_len) containing character indices.
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, num_labels)
        """
        # 1. Embed: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        emb = self.embedding(x)
        # 2. Permute for conv1d: (batch, embedding_dim, seq_len)
        emb = emb.transpose(1, 2)

        # 3. Apply each convolution + activation
        conv_outputs = []
        for conv in self.convs:
            out = conv(emb)              # (batch, num_filters, seq_len)
            out = F.relu(out)
            conv_outputs.append(out)

        # 4. Concatenate on the channel dimension
        conv_cat = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes), seq_len)

        # 5. Dropout
        dropped = self.dropout(conv_cat)

        # 6. Project to label space
        logits = self.classifier(dropped)  # (batch, num_labels, seq_len)

        # 7. Permute back: (batch, seq_len, num_labels)
        logits = logits.transpose(1, 2)
        return logits
