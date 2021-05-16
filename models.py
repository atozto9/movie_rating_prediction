import torch

from torch import nn


class MovieRatingModel(nn.Module):

    def __init__(self, symbol_size):
        super(MovieRatingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=symbol_size, embedding_dim=128)

        self.conv_1 = nn.Sequential(nn.Conv1d(128, 128, 3, stride=1, padding=2,bias=True),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Dropout(0.2))
        self.conv_2 = nn.Sequential(nn.Conv1d(128, 64, 3, stride=1, padding=2,bias=True),
                               nn.BatchNorm1d(64),
                               nn.ReLU(),
                               nn.Dropout(0.2))
        self.conv_3 = nn.Sequential(nn.Conv1d(64, 128, 3, stride=1, padding=2,bias=True),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                               nn.Dropout(0.2))

        self.blstm = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)

        self.fc_1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

        self.fc_coarse = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x, ilens):

        x = self.embedding(x)

        x = x.transpose(1, 2)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, ilens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.blstm(x)
        x, hlens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x[range(hlens.size(0)), hlens - 1, :]

        x = self.fc_1(x)
        x = self.relu(x)

        coarse_x = self.fc_coarse(x)
        rating_x = self.fc(x)

        return rating_x, coarse_x