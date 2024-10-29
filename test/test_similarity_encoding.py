from . import left, right, sencoder


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
