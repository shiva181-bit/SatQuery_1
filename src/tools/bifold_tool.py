from pathlib import Path

import numpy as np

from ..data.loader import (
    BIFOLD_RESNET50_ALL_V011_CHANNELS,
    bigearthnet_14_to_bifold_v011,
    inspect_bigearthnet_14_patch,
    inspect_pair,
    load_bigearthnet_14_patch,
    normalize_bifold_v011,
)
from ..models.bifold import BIFOLDModel


# ConfigILM NEW_LABELS is lexicographically sorted, which is the ordering used
# by the published checkpoint's default multi-hot targets.
CLASS_NAMES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands",
    "Broad-leaved forest", "Coastal wetlands", "Complex cultivation patterns",
    "Coniferous forest", "Industrial or commercial units", "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest", "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures", "Permanent crops",
    "Transitional woodland, shrub", "Urban fabric",
]


class BIFOLDTool:
    ID = "TOOL_B_CROSS_MODAL_S1_S2"
    CHECKPOINT_PROFILE = "BIFOLD BigEarthNet v2 ResNet-50 all v0.1.1"

    def __init__(self, checkpoint: str | Path):
        self.model = BIFOLDModel(checkpoint)

    def run(self, s1_path: str | Path, s2_path: str | Path) -> dict:
        """Reject unlabelled GeoTIFFs instead of guessing their band order."""
        return {
            "tool": self.ID,
            "status": "DATA_NOT_READY",
            "message": (
                "Tool B requires a documented BigEarthNet_20 14-channel patch "
                "or explicit verified channel metadata; raw GeoTIFF channel count "
                "alone is not sufficient for this legacy checkpoint."
            ),
            "input": inspect_pair(s1_path, s2_path),
        }

    def run_bigearthnet_patch(self, patch_dir: str | Path) -> dict:
        info = inspect_bigearthnet_14_patch(patch_dir)
        source = load_bigearthnet_14_patch(patch_dir)
        x = normalize_bifold_v011(bigearthnet_14_to_bifold_v011(source))
        scores = self.model.predict_normalized(x)
        return {
            "tool": self.ID,
            "status": "SUCCESS",
            "input": info,
            "checkpoint_profile": self.CHECKPOINT_PROFILE,
            "normalization": "BigEarthNet v2 official-train 120_nearest mean/std",
            "channel_count": int(x.shape[0]),
            "channel_order": list(BIFOLD_RESNET50_ALL_V011_CHANNELS),
            "top_classes": self._top_classes(scores["class_scores"]),
            "device": scores["device"],
        }

    @staticmethod
    def _top_classes(scores: list[float], k: int = 5) -> list[dict]:
        indices = np.argsort(scores)[::-1][:k]
        return [
            {"class_id": int(index), "label": CLASS_NAMES[index],
             "score": round(float(scores[index]), 4)}
            for index in indices
        ]
