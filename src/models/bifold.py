from pathlib import Path

import numpy as np
import torch
import timm


class BIFOLDModel:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "resnet50",
            pretrained=False,
            in_chans=12,
            num_classes=19,
        )

        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_normalized(self, x: np.ndarray) -> dict:
        if x.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got {x.shape}")

        if x.shape[0] != 12:
            raise ValueError(
                f"BIFOLD model requires 12 channels, got {x.shape[0]}"
            )

        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits)[0]

        return {
            "class_scores": probabilities.cpu().numpy().tolist(),
            "device": self.device,
        }

    def predict(self, x: np.ndarray) -> dict:
        from ..data.loader import normalize_bifold

        return self.predict_normalized(
            normalize_bifold(x)
        )