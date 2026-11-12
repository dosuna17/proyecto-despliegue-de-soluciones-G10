"""
TinySleepNet sin la LSTM (ablacion): cada epoca se clasifica sola, con la misma CNN y una capa lineal.

Sirve para medir cuanto aporta el contexto de las epocas vecinas. Todo lo demas (entrada flexible
de canales, hiperparametros, entrenamiento) es igual al modelo flexible.

Entrada y salida con el mismo formato que TinySleepNet con LSTM, para usar el mismo bucle de
entrenamiento: x (batch, secuencia, canales, 3000) -> (logits (batch, secuencia, 5), None).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from .tinysleepnet import TinySleepNet, TinySleepNetConfig


class TinySleepNetSinLSTM(TinySleepNet):
    def __init__(self, config: TinySleepNetConfig):
        super().__init__(dataclasses.replace(config, use_rnn=False))

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        batch, seq_len, canales, n_muestras = x.shape
        feats = self.cnn(x.reshape(batch * seq_len, canales, n_muestras))      # una epoca a la vez
        logits = self.fc(feats).view(batch, seq_len, -1)
        return logits, None

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        self.eval()
        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)
        return {"pred_class": pred_class, "confidence": confidence, "probabilities": probs}, None
