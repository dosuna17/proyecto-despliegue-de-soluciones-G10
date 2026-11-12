"""
TinySleepNet con EMG en una rama aparte (la CNN no ve el EMG).

El EMG submental viene a 1 Hz: 30 valores por epoca de 30 s (las senales de
EEG/EOG tienen 3000). Para que todas las posiciones tengan el mismo largo, el
EMG viaja en el tensor de entrada repetido 100 veces (30 x 100 = 3000); la rama
recupera los 30 valores originales con x[..., ::100].

Entrada: (batch, secuencia, 8, 3000)
    canales 0-3 : senales Fpz-Cz, Pz-Oz, EOG, EMG
    canales 4-7 : marcas de presencia (0 = ausente, 1 = presente) de cada una

Camino de los datos:
    canales [0, 1, 2, 4, 5, 6]  --> CNN  --> 2048 numeros por epoca ----+
                                                                         +--> concatenar (2048 + 16)
    EMG (30 valores) + marca    --> rama EMG ('mlp' o 'conv') --> 16 ---+        |
                                                                                 v
                                                                   LSTM --> capa lineal --> 5 clases

Dos variantes de la rama, con tamano parecido (alrededor de 1.5 mil parametros):
    'mlp'  : Linear(31, 32) - ReLU - Linear(32, 16) - ReLU
    'conv' : Conv1d(2, 16, 5) - ReLU - Conv1d(16, 16, 5) - ReLU - promedio sobre los 30 segundos
             (2 canales de entrada = los 30 valores y la marca repetida a lo largo de los 30)
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn

from .tinysleepnet import TinySleepNet, TinySleepNetConfig

REPETIR = 100                      # 30 valores -> 3000 muestras
N_VALORES = 30
EMB_EMG = 16
IDX_CNN = (0, 1, 2, 4, 5, 6)       # 3 senales de 100 Hz + sus 3 marcas
IDX_EMG, IDX_MARCA_EMG = 3, 7
N_CANALES_ENTRADA_EMG = 8
RAMAS = ("mlp", "conv")


class RamaEMGMLP(nn.Module):
    def __init__(self, n_valores: int = N_VALORES, salida: int = EMB_EMG):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_valores + 1, 32), nn.ReLU(),
            nn.Linear(32, salida), nn.ReLU(),
        )

    def forward(self, valores: torch.Tensor, marca: torch.Tensor) -> torch.Tensor:
        # valores: (N, 30), marca: (N, 1) -> (N, 16)
        return self.red(torch.cat([valores, marca], dim=-1))


class RamaEMGConv(nn.Module):
    def __init__(self, salida: int = EMB_EMG):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv1d(2, salida, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(salida, salida, kernel_size=5, padding=2), nn.ReLU(),
        )

    def forward(self, valores: torch.Tensor, marca: torch.Tensor) -> torch.Tensor:
        # valores: (N, 30), marca: (N, 1) -> (N, 16)
        x = torch.stack([valores, marca.expand_as(valores)], dim=1)   # (N, 2, 30)
        return self.red(x).mean(dim=-1)


class TinySleepNetEMG(TinySleepNet):
    """TinySleepNet cuyo EMG entra por una rama aparte que se une a la salida de la CNN."""

    def __init__(self, config: TinySleepNetConfig, rama: str = "mlp"):
        if rama not in RAMAS:
            raise ValueError(f"rama debe ser una de {RAMAS}; llego '{rama}'.")
        if not config.use_rnn:
            raise ValueError("TinySleepNetEMG necesita la LSTM (use_rnn=True).")
        # La CNN solo ve las 3 posiciones de 100 Hz con sus marcas (6 canales).
        super().__init__(dataclasses.replace(config, input_channels=len(IDX_CNN)))
        self.rama = rama
        self.emg_branch = RamaEMGMLP() if rama == "mlp" else RamaEMGConv()
        # La LSTM recibe lo de la CNN mas los 16 numeros de la rama del EMG.
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim + EMB_EMG, hidden_size=config.n_rnn_units,
            num_layers=config.n_rnn_layers, batch_first=True,
            dropout=config.rnn_dropout if config.n_rnn_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        # x: (batch, seq, 8, 3000)
        batch, seq_len, canales, n_muestras = x.shape
        if canales != N_CANALES_ENTRADA_EMG:
            raise ValueError(f"Se esperaban {N_CANALES_ENTRADA_EMG} canales de entrada; llegaron {canales}.")
        cnn_in = x[:, :, list(IDX_CNN), :].reshape(batch * seq_len, len(IDX_CNN), n_muestras)
        feats = self.cnn(cnn_in).view(batch, seq_len, self.cnn_out_dim)
        valores = x[:, :, IDX_EMG, ::REPETIR].reshape(batch * seq_len, N_VALORES)
        marca = x[:, :, IDX_MARCA_EMG, :1].reshape(batch * seq_len, 1)
        emb = self.emg_branch(valores, marca).view(batch, seq_len, EMB_EMG)
        lstm_out, hidden = self.lstm(torch.cat([feats, emb], dim=-1), hidden)
        lstm_out = self.rnn_dropout(lstm_out)
        return self.fc(lstm_out), hidden
