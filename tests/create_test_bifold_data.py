from pathlib import Path

import numpy as np
import rasterio

OUT = Path("data/uploads")
OUT.mkdir(parents=True, exist_ok=True)

H, W = 120, 120

rng = np.random.default_rng(42)

s1 = rng.normal(-17, 7, (2, H, W)).astype(np.float32)
s2 = rng.uniform(200, 4500, (10, H, W)).astype(np.float32)


def save(path, data):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=data.shape[0],
        dtype="float32",
    ) as dst:
        dst.write(data)


save(OUT / "test_s1_12ch.tif", s1)
save(OUT / "test_s2_12ch.tif", s2)

print("Created:")
print(OUT / "test_s1_12ch.tif")
print(OUT / "test_s2_12ch.tif")