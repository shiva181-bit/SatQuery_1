from pathlib import Path

import numpy as np
import rasterio


class ChangeTool:
    ID = "TOOL_C_BITEMPORAL_CHANGE"

    def run(self, image_t1: str | Path, image_t2: str | Path) -> dict:
        t1 = self._read(image_t1)
        t2 = self._read(image_t2)

        if t1.shape != t2.shape:
            raise ValueError(
                "Temporal images must have identical dimensions and band count: "
                f"{t1.shape} vs {t2.shape}"
            )

        # Normalize T1 and T2 using shared statistics so that the
        # comparison is less sensitive to different absolute ranges.
        t1_norm, t2_norm = self._normalize_pair(t1, t2)

        diff = np.abs(t2_norm - t1_norm)

        # Collapse multiple bands into one change-intensity image.
        change_map = np.mean(diff, axis=0)

        mean_change = float(change_map.mean())
        max_change = float(change_map.max())

        threshold = self._calculate_threshold(change_map)

        changed_pixels = change_map > threshold
        changed_fraction = float(changed_pixels.mean())

        observations = [
            f"Compared {t1.shape[0]} band(s) across two temporal observations.",
            "Pixel differences were computed after shared-range normalization.",
            "The reported change represents apparent image-level change, not semantic object change.",
        ]

        if changed_fraction == 0:
            observations.append("No pixels exceeded the change threshold.")
        elif changed_fraction < 0.10:
            observations.append("Only a small portion of pixels exceeded the threshold.")
        elif changed_fraction < 0.30:
            observations.append("A moderate portion of pixels exceeded the threshold.")
        else:
            observations.append("A substantial portion of pixels exceeded the threshold.")

        return {
            "status": "SUCCESS",
            "tool": self.ID,
            "answer": self._narrative(changed_fraction, mean_change),
            "observations": observations,
            "metrics": {
                "image_shape": [
                    int(t1.shape[1]),
                    int(t1.shape[2]),
                ],
                "bands_compared": int(t1.shape[0]),
                "mean_change": mean_change,
                "max_change": max_change,
                "changed_pixels": int(changed_pixels.sum()),
                "total_pixels": int(changed_pixels.size),
                "changed_fraction": changed_fraction,
                "changed_percentage": round(changed_fraction * 100, 2),
                "threshold": float(threshold),
            },
        }

    @staticmethod
    def _read(path: str | Path) -> np.ndarray:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)

        if data.size == 0:
            raise ValueError(f"Image contains no readable pixels: {path}")

        if not np.isfinite(data).any():
            raise ValueError(f"Image contains no finite pixel values: {path}")

        # Replace invalid values with NaN. They will be handled during
        # normalization.
        data[~np.isfinite(data)] = np.nan

        return data

    @staticmethod
    def _normalize_pair(
        t1: np.ndarray,
        t2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        bands = t1.shape[0]

        t1_norm = np.empty_like(t1, dtype=np.float32)
        t2_norm = np.empty_like(t2, dtype=np.float32)

        for band in range(bands):
            combined = np.concatenate(
                [
                    t1[band].reshape(-1),
                    t2[band].reshape(-1),
                ]
            )

            valid = combined[np.isfinite(combined)]

            if valid.size == 0:
                t1_norm[band] = 0.0
                t2_norm[band] = 0.0
                continue

            low, high = np.percentile(valid, [2, 98])

            # Avoid division by zero for constant bands.
            if high <= low:
                low = float(valid.min())
                high = float(valid.max())

            if high <= low:
                t1_norm[band] = 0.0
                t2_norm[band] = 0.0
                continue

            t1_norm[band] = np.clip(
                (np.nan_to_num(t1[band], nan=low) - low)
                / (high - low),
                0.0,
                1.0,
            )

            t2_norm[band] = np.clip(
                (np.nan_to_num(t2[band], nan=low) - low)
                / (high - low),
                0.0,
                1.0,
            )

        return t1_norm, t2_norm

    @staticmethod
    def _calculate_threshold(change_map: np.ndarray) -> float:
        valid = change_map[np.isfinite(change_map)]

        if valid.size == 0:
            return 0.0

        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))

        # Robust threshold based on median + 3 MAD.
        threshold = median + (3.0 * mad)

        # If the image has almost no variation in its difference map,
        # use a high percentile as a fallback.
        if mad < 1e-8:
            threshold = float(np.percentile(valid, 95))

        return threshold

    @staticmethod
    def _narrative(fraction: float, mean_change: float) -> str:
        percentage = fraction * 100

        if fraction < 0.10:
            level = "low"
        elif fraction < 0.30:
            level = "moderate"
        else:
            level = "high"

        return (
            f"The two temporal observations show {level} apparent change: "
            f"approximately {percentage:.2f}% of pixels exceeded the "
            f"change threshold. Mean normalized change intensity was "
            f"{mean_change:.4f}."
        )