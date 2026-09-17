from pathlib import Path

import numpy as np
import rasterio


BIFOLD_CHANNELS = 12

# Source order recorded in BigEarthNet_20/*/patch_info.txt.
BIGEARTHNET_14_CHANNELS = (
    "VV", "VH", "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B11", "B12",
)

# This is deliberately *not* the current BigEarthNet v2 standard order. The
# local checkpoint is the v0.1.1 "all" model, whose card documents this order.
BIFOLD_RESNET50_ALL_V011_CHANNELS = (
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12",
    "B8A", "VH", "VV",
)

# BigEarthNet v2 official-train statistics for 120 px nearest-neighbour input.
# Values are in the checkpoint's v0.1.1 channel order.
_BEN_V2_120_NEAREST_MEAN = {
    "B02": 438.3720703125, "B03": 614.0556640625,
    "B04": 588.4096069335938, "B05": 942.8433227539062,
    "B06": 1769.931640625, "B07": 2049.551513671875,
    "B08": 2193.2919921875, "B8A": 2235.556640625,
    "B11": 1568.226806640625, "B12": 997.7324829101562,
    "VH": -19.352558135986328, "VV": -12.643863677978516,
}
_BEN_V2_120_NEAREST_STD = {
    "B02": 607.02685546875, "B03": 603.2968139648438,
    "B04": 684.56884765625, "B05": 738.4326782226562,
    "B06": 1100.4560546875, "B07": 1275.805419921875,
    "B08": 1369.3717041015625, "B8A": 1356.5440673828125,
    "B11": 1070.1612548828125, "B12": 813.5276489257812,
    "VH": 5.590505599975586, "VV": 5.133493900299072,
}


def read_bands(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def inspect_pair(s1_path: str | Path, s2_path: str | Path) -> dict:
    s1 = read_bands(s1_path)
    s2 = read_bands(s2_path)
    return {
        "s1_shape": list(s1.shape),
        "s2_shape": list(s2.shape),
        "combined_channels": int(s1.shape[0] + s2.shape[0]),
        "spatially_aligned": s1.shape[1:] == s2.shape[1:],
        "native_bifold_ready": False,
        "reason": (
            "Standalone GeoTIFF bands have no verified band-order metadata. "
            "Use a documented BigEarthNet_20 14_channel_data.npy patch, or "
            "add explicit band metadata before inference."
        ),
    }


def load_bigearthnet_14_patch(patch_dir: str | Path) -> np.ndarray:
    patch_dir = Path(patch_dir)
    array_path = patch_dir / "14_channel_data.npy"
    info_path = patch_dir / "patch_info.txt"
    if not array_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(
            "A BigEarthNet patch must contain 14_channel_data.npy and "
            "patch_info.txt."
        )

    metadata = info_path.read_text(encoding="utf-8")
    expected_order = "\n".join(
        f"{index}: {band}" for index, band in enumerate(BIGEARTHNET_14_CHANNELS)
    )
    if "Number of channels: 14" not in metadata or expected_order not in metadata:
        raise ValueError(
            "patch_info.txt does not document the expected 14-channel "
            "BigEarthNet band order."
        )

    data = np.load(array_path).astype(np.float32)
    if data.ndim != 3 or data.shape[0] != len(BIGEARTHNET_14_CHANNELS):
        raise ValueError(
            "Expected BigEarthNet data shaped [14, height, width], got "
            f"{data.shape}."
        )
    if data.shape[1:] != (120, 120):
        raise ValueError(
            "The local checkpoint expects 120x120 patches, got "
            f"{data.shape[1:]}."
        )
    if not np.isfinite(data).all():
        raise ValueError("BigEarthNet patch contains non-finite pixel values.")
    return data


def inspect_bigearthnet_14_patch(patch_dir: str | Path) -> dict:
    data = load_bigearthnet_14_patch(patch_dir)
    return {
        "source_format": "BigEarthNet_20 14_channel_data.npy",
        "source_shape": [int(value) for value in data.shape],
        "source_channel_order": list(BIGEARTHNET_14_CHANNELS),
        "dropped_bands": ["B01", "B09"],
        "model_channel_order": list(BIFOLD_RESNET50_ALL_V011_CHANNELS),
        "native_bifold_ready": True,
    }


def bigearthnet_14_to_bifold_v011(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3 or data.shape[0] != len(BIGEARTHNET_14_CHANNELS):
        raise ValueError(f"Expected [14, H, W], got {data.shape}")
    index = {band: position for position, band in enumerate(BIGEARTHNET_14_CHANNELS)}
    return np.stack(
        [data[index[band]] for band in BIFOLD_RESNET50_ALL_V011_CHANNELS], axis=0
    ).astype(np.float32, copy=False)


def normalize_bifold_v011(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[0] != BIFOLD_CHANNELS:
        raise ValueError(f"BIFOLD v0.1.1 requires [12, H, W], got {x.shape}")
    mean = np.asarray(
        [_BEN_V2_120_NEAREST_MEAN[band] for band in BIFOLD_RESNET50_ALL_V011_CHANNELS],
        dtype=np.float32,
    )[:, None, None]
    std = np.asarray(
        [_BEN_V2_120_NEAREST_STD[band] for band in BIFOLD_RESNET50_ALL_V011_CHANNELS],
        dtype=np.float32,
    )[:, None, None]
    return (x.astype(np.float32, copy=False) - mean) / std
