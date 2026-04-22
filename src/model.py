import os
import torch
import torch.nn as nn
import config


class BatteryTemperatureLSTM(nn.Module):
    def __init__(self, input_size=config.input_size, hidden_size=config.hidden_size, num_layers=config.num_layers,dropout=config.dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out=self.dropout(out[:,-1,:])
        out = self.linear(out)
        return out
