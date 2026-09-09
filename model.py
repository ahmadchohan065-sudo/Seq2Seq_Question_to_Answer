import random
import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3

class Encoder(nn.Module):
    def __init__(self, vocab_size=8000, emb_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2,
                             batch_first=True, bidirectional=True, dropout=0.3)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        cell_cat = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden_proj = torch.tanh(self.fc_hidden(hidden_cat))
        cell_proj = torch.tanh(self.fc_cell(cell_cat))

        return outputs, hidden_proj, cell_proj

