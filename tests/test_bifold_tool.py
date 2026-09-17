from src.data.loader import inspect_pair


def test_unlabelled_geotiffs_are_not_treated_as_checkpoint_ready():
    result = inspect_pair(
        "data/uploads/test_s1_12ch.tif",
        "data/uploads/test_s2_12ch.tif",
    )

    assert result["native_bifold_ready"] is False
    assert "verified band-order metadata" in result["reason"]
