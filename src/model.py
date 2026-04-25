import torch
import torch.nn as nn

import config


class BatteryTemperatureLSTM(nn.Module):
    def __init__(
        self,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        ocv_hidden_size=config.ocv_hidden_size,
        ocv_min=config.ocv_min,
        ocv_max=config.ocv_max,
    ):
        super().__init__()

        self.ocv_min = float(ocv_min)
        self.ocv_max = float(ocv_max)

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.temperature_head = nn.Linear(hidden_size, 1)
        self.ocv_head = nn.Sequential(
            nn.Linear(2, ocv_hidden_size),
            nn.ReLU(),
            nn.Linear(ocv_hidden_size, ocv_hidden_size),
            nn.ReLU(),
            nn.Linear(ocv_hidden_size, 1),
        )

        self.lambda_1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda_2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x, soc=None, soh=None, return_aux=False):
        out, _ = self.lstm(x)
        hidden = self.dropout(out[:, -1, :])
        temperature = self.temperature_head(hidden)

        if not return_aux:
            return temperature

        if soc is None or soh is None:
            raise ValueError('return_aux=True 时必须提供 soc 和 soh')

        ocv_input = torch.cat([soc.view(-1, 1), soh.view(-1, 1)], dim=1)
        ocv_raw = self.ocv_head(ocv_input)
        ocv = self.ocv_min + (self.ocv_max - self.ocv_min) * torch.sigmoid(ocv_raw)
        return temperature, ocv
