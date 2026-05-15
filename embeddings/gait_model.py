# gait_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 128,
        emb_dim: int = 128,
        num_layers: int = 1,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (B, T, input_dim)
        )
        self.fc_emb = nn.Linear(hidden_dim, emb_dim)
        self.num_classes = num_classes
        if num_classes is not None:
            self.fc_cls = nn.Linear(emb_dim, num_classes)
        else:
            self.fc_cls = None

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, input_dim)
        return:
          emb: (B, emb_dim)
          logits: (B, num_classes) или None
        """
        out, (h_n, c_n) = self.lstm(x)   # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]                 # (B, hidden_dim)
        emb = self.fc_emb(h_last)        # (B, emb_dim)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.fc_cls(emb) if self.fc_cls is not None else None
        return emb, logits