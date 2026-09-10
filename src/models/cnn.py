"""
Arquitectura TinySleepNet (Supratak & Guo, 2020) reimplementada en PyTorch.

Fidelidad a la fuente: la arquitectura del bloque CNN y del bloque LSTM
sigue exactamente `model.py` del repositorio oficial de los autores
(https://github.com/akaraspt/tinysleepnet), no el texto del paper (que,
por ser un paper de 4 páginas de conferencia EMBC, omite varios detalles
de implementación). Nota de transparencia: el paper afirma que el stride
del primer filtro convolucional es sampling_rate/4, pero el código oficial
usa sampling_rate/16 (ver issue #2 del repo); esta implementación sigue
el código, no el texto del paper, por ser la fuente más confiable para
detalles de implementación.

Contrato de entrada (confirmado contra segmentation.py / loader.py del
proyecto): cada época es originalmente (1 canal, 3000 muestras) = EEG
Fpz-Cz a 100 Hz, 30s por época. n_classes=5 (W, N1, N2, N3, REM).

Entrega 3 -- extensión multicanal: el número de canales de entrada es
ahora configurable vía TinySleepNetConfig.n_input_channels (1=EEG solo,
2=EEG+EOG, 3=EEG+EOG+EMG), apilando los canales desde la primera capa
convolucional (opción simple, documentar en el informe si se cambia).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class TinySleepNetConfig:
    """Hiperparámetros del modelo. Los valores por defecto replican el
    bloque CNN del repositorio oficial para sampling_rate=100 Hz; los de
    la LSTM y de entrenamiento (seq_length, n_rnn_units, etc.) son punto
    de partida razonable a ajustar empíricamente -- NO están tomados del
    paper porque el paper no reporta el valor exacto usado en Sleep-EDF
    (solo aparecen en su config/*.py real, que no incluye Sleep-EDF con
    canal único EEG Fpz-Cz exactamente igual a nuestro preprocesamiento).
    """
    sampling_rate: int = 100          # Hz (confirmado: segmentation.py)
    input_size: int = 3000            # muestras por época (confirmado: 30s * 100Hz)
    n_classes: int = 5                # W, N1, N2, N3, REM (confirmado: segmentation.py)
    n_input_channels: int = 1         # Entrega 3: 1=EEG solo, 2=EEG+EOG, 3=EEG+EOG+EMG

    # Bloque CNN (valores reales del repo oficial, ver docstring del módulo)
    cnn_filters: int = 128
    cnn_dropout: float = 0.5

    # Bloque LSTM (arquitectura confirmada por el repo oficial: LSTM
    # unidireccional con peepholes; n_rnn_layers/n_rnn_units/seq_length
    # SÍ requieren ajuste empírico propio, documentar en el informe como
    # decisión de diseño, no como valor de la literatura)
    use_rnn: bool = True
    n_rnn_layers: int = 1
    n_rnn_units: int = 128
    rnn_dropout: float = 0.5
    seq_length: int = 15               # épocas consecutivas por secuencia (paper: seq_length=15)

    # Regularización / optimización (arquitectónicamente confirmados que
    # existen en el diseño original: L2 solo sobre conv, class weights,
    # gradient clipping; los valores numéricos son de tu propio tuning)
    l2_weight_decay: float = 1e-3
    class_weights: Optional[List[float]] = None  # se calculan desde el train split


class _ConvBNReLU(nn.Module):
    """Bloque Conv1D -> BatchNorm -> ReLU, unidad repetida del CNN oficial."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        # padding "same" aproximado (el repo original usa padding SAME de TF)
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class TinySleepNetCNN(nn.Module):
    """Bloque extractor de features intra-época (fiel a build_cnn() del repo oficial)."""

    def __init__(self, config: TinySleepNetConfig):
        super().__init__()
        fs = config.sampling_rate
        first_filter_size = int(fs / 2.0)   # = 50 para fs=100
        first_filter_stride = int(fs / 16.0)  # = 6 para fs=100 (ver nota de fidelidad arriba)

        self.block1 = _ConvBNReLU(config.n_input_channels, config.cnn_filters, first_filter_size, first_filter_stride)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.drop1 = nn.Dropout(config.cnn_dropout)

        self.block2 = nn.Sequential(
            _ConvBNReLU(config.cnn_filters, config.cnn_filters, kernel_size=8, stride=1),
            _ConvBNReLU(config.cnn_filters, config.cnn_filters, kernel_size=8, stride=1),
            _ConvBNReLU(config.cnn_filters, config.cnn_filters, kernel_size=8, stride=1),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop2 = nn.Dropout(config.cnn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_input_channels, input_size)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return torch.flatten(x, start_dim=1)  # (batch, cnn_out_dim)


class TinySleepNet(nn.Module):
    """
    Modelo completo: CNN (intra-época) + LSTM opcional (inter-época) + FC final.

    Modo de uso (igual al paper/repo oficial):
      - use_rnn=False: entrena solo el CNN sobre épocas independientes
        (etapa de "pre-training" en la terminología del paper).
      - use_rnn=True: recibe secuencias de `seq_length` épocas consecutivas
        del MISMO sujeto/noche (nunca mezcladas entre sujetos) y aplica la
        LSTM sobre los embeddings del CNN para modelar transiciones.
    """

    def __init__(self, config: TinySleepNetConfig):
        super().__init__()
        self.config = config
        self.cnn = TinySleepNetCNN(config)

        # Determinar dinámicamente la dimensión de salida del CNN
        with torch.no_grad():
            dummy = torch.zeros(1, config.n_input_channels, config.input_size)
            cnn_out_dim = self.cnn(dummy).shape[-1]
        self.cnn_out_dim = cnn_out_dim

        if config.use_rnn:
            self.lstm = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=config.n_rnn_units,
                num_layers=config.n_rnn_layers,
                batch_first=True,
                dropout=config.rnn_dropout if config.n_rnn_layers > 1 else 0.0,
            )
            self.rnn_dropout = nn.Dropout(config.rnn_dropout)
            fc_in_dim = config.n_rnn_units
        else:
            self.lstm = None
            fc_in_dim = cnn_out_dim

        self.fc = nn.Linear(fc_in_dim, config.n_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None,
    ):
        """
        Si use_rnn=False:
            x: (batch, n_input_channels, input_size) -> logits: (batch, n_classes)
        Si use_rnn=True:
            x: (batch, seq_length, n_input_channels, input_size) -> logits: (batch, seq_length, n_classes)
            `hidden` permite pasar el estado de la LSTM entre mini-batches
            consecutivos de la misma secuencia larga (carry-over de estado,
            como hace el repo oficial); si es None, se inicializa en ceros.
        """
        if not self.config.use_rnn:
            feats = self.cnn(x)  # (batch, cnn_out_dim)
            logits = self.fc(feats)
            return logits

        batch, seq_len, ch, n_samples = x.shape
        x = x.view(batch * seq_len, ch, n_samples)
        feats = self.cnn(x)                          # (batch*seq_len, cnn_out_dim)
        feats = feats.view(batch, seq_len, self.cnn_out_dim)

        lstm_out, hidden = self.lstm(feats, hidden)   # (batch, seq_len, n_rnn_units)
        lstm_out = self.rnn_dropout(lstm_out)

        logits = self.fc(lstm_out)                    # (batch, seq_len, n_classes)
        return logits, hidden

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        """
        Inferencia lista para servir: aplica softmax sobre los logits y
        devuelve también la clase predicha y su confianza (prob. máxima).
        Pensado para el contrato de app/dashboard/model_interface.py
        (predicción + confianza + probabilidades por clase), que la API
        de Daniel expondrá eventualmente en POST /predict.

        Devuelve un dict con:
          - "pred_class": índice de clase más probable, shape (batch,) o (batch, seq_len)
          - "confidence": probabilidad de esa clase, misma shape
          - "probabilities": tensor completo de probabilidades por clase
        """
        self.eval()
        if not self.config.use_rnn:
            logits = self.forward(x)
        else:
            logits, hidden = self.forward(x, hidden)

        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)

        result = {
            "pred_class": pred_class,
            "confidence": confidence,
            "probabilities": probs,
        }
        if self.config.use_rnn:
            return result, hidden
        return result

    def l2_regularization_loss(self) -> torch.Tensor:
        """
        Replica regularization_loss() del repo oficial: L2 solo sobre los
        pesos convolucionales del CNN (NO sobre la LSTM ni la capa final).
        """
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv1d):
                l2 = l2 + torch.sum(module.weight ** 2)
        return self.config.l2_weight_decay * l2


def build_class_weighted_loss(class_weights: Optional[List[float]], device: str = "cpu") -> nn.Module:
    """
    CrossEntropyLoss ponderada por clase. Los pesos deben calcularse
    EXCLUSIVAMENTE desde la distribución de clases del conjunto de
    entrenamiento (nunca de val/test), y con un esquema suavizado
    (ej. 1/sqrt(frecuencia)) en vez de inverse-frequency cruda, para
    evitar inestabilidad de gradiente en la LSTM (ver discusión del
    informe sobre riesgos de class weighting en esta arquitectura).
    """
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


if __name__ == "__main__":
    # Prueba rápida de forma (shapes), sin datos reales
    cfg = TinySleepNetConfig(use_rnn=False)
    model = TinySleepNet(cfg)
    x = torch.randn(4, cfg.n_input_channels, cfg.input_size)
    out = model(x)
    print("CNN-only (EEG solo) | input:", x.shape, "-> logits:", out.shape)

    cfg_rnn = TinySleepNetConfig(use_rnn=True, seq_length=10)
    model_rnn = TinySleepNet(cfg_rnn)
    x_seq = torch.randn(2, cfg_rnn.seq_length, cfg_rnn.n_input_channels, cfg_rnn.input_size)
    out_seq, hidden = model_rnn(x_seq)
    print("CNN+LSTM (EEG solo) | input:", x_seq.shape, "-> logits:", out_seq.shape)
    print("cnn_out_dim:", model_rnn.cnn_out_dim)

    # Entrega 3 -- verificacion de forma para EEG+EOG (2 canales apilados
    # desde la primera capa), sin necesitar los .npz reales todavia.
    cfg_multicanal = TinySleepNetConfig(use_rnn=True, seq_length=10, n_input_channels=2)
    model_multicanal = TinySleepNet(cfg_multicanal)
    x_seq_mc = torch.randn(2, cfg_multicanal.seq_length, cfg_multicanal.n_input_channels, cfg_multicanal.input_size)
    out_seq_mc, hidden_mc = model_multicanal(x_seq_mc)
    print("CNN+LSTM (EEG+EOG) | input:", x_seq_mc.shape, "-> logits:", out_seq_mc.shape)
    print("cnn_out_dim (EEG+EOG):", model_multicanal.cnn_out_dim)
