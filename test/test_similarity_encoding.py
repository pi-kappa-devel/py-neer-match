from . import left, left_short, right, right_short, sencoder
import pandas as pd


def test_encoding():
    encoded = sencoder(left, right.iloc[:-3, :])
    assert sencoder.similarity_map.no_associations() == len(
        encoded
    ), f"Expected {sencoder.similarity_map.no_associations()}, got {len(encoded)}"
    for i, v in enumerate(encoded):
        assert v.shape[-1] == sencoder.similarity_map.association_sizes()[i], (
            f"Expected {sencoder.similarity_map.association_sizes()[i]}, "
            f"got {v.shape[-1]}"
        )


def test_encoding_report():
    report = sencoder.report_encoding(left_short, right_short)
    assert isinstance(report, list), "Failed to report encoding."
    expected_shape = (len(sencoder.similarity_map), len(left_short))
    for item in report:
        assert isinstance(item, pd.DataFrame), "Failed to report encoding."
        assert item.shape == expected_shape, (
            "Unexpected report shape. " f"Expected: {expected_shape}, got: {item.shape}"
        )
