import torch.nn as nn

class KoreanSpacingLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300, num_labels=2):
        super(KoreanSpacingLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim*2, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits