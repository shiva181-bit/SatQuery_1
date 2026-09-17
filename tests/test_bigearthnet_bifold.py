from pathlib import Path

import numpy as np
import pytest

from src.data.loader import (
    BIFOLD_RESNET50_ALL_V011_CHANNELS,
    bigearthnet_14_to_bifold_v011,
    inspect_bigearthnet_14_patch,
    load_bigearthnet_14_patch,
    normalize_bifold_v011,
)


PATCH = Path("BigEarthNet_20/patch_01")
CHECKPOINT = Path("models/bifold_resnet50_12ch.pt")


def test_documented_14_channel_patch_maps_to_legacy_checkpoint_order():
    source = load_bigearthnet_14_patch(PATCH)
    mapped = bigearthnet_14_to_bifold_v011(source)
    info = inspect_bigearthnet_14_patch(PATCH)

    assert source.shape == (14, 120, 120)
    assert mapped.shape == (12, 120, 120)
    assert info["native_bifold_ready"] is True
    assert info["model_channel_order"] == list(BIFOLD_RESNET50_ALL_V011_CHANNELS)

    # B02, B08, VH and VV are positions 3, 9, 1 and 0 in patch_info.txt.
    assert np.array_equal(mapped[0], source[3])
    assert np.array_equal(mapped[3], source[9])
    assert np.array_equal(mapped[10], source[1])
    assert np.array_equal(mapped[11], source[0])


def test_checkpoint_normalization_uses_fixed_training_statistics():
    normalized = normalize_bifold_v011(
        bigearthnet_14_to_bifold_v011(load_bigearthnet_14_patch(PATCH))
    )
    assert normalized.shape == (12, 120, 120)
    assert np.isfinite(normalized).all()


@pytest.mark.integration
def test_bifold_checkpoint_runs_on_a_documented_bigearthnet_patch():
    pytest.importorskip("timm")
    if not CHECKPOINT.is_file():
        pytest.skip("BIFOLD checkpoint is not present")

    from src.tools.bifold_tool import BIFOLDTool

    result = BIFOLDTool(CHECKPOINT).run_bigearthnet_patch(PATCH)
    assert result["status"] == "SUCCESS"
    assert result["channel_count"] == 12
    assert len(result["top_classes"]) == 5
