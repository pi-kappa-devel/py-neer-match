"""The faster encoding and labelling produce exactly what the previous code did."""

from . import left, matches, right, sencoder, smap
from neer_match.data_generator import DataGenerator
from neer_match.similarity_encoding import SimilarityEncoder
from neer_match.similarity_map import SimilarityMap
import numpy as np
import pandas as pd


def reference_encoding(encoder, left, right):
    """The previous SimilarityEncoder.encode_as_matrix: one iloc lookup per value."""
    lx = left[encoder.similarity_map.lcols]
    rx = right[encoder.similarity_map.rcols]
    return np.array(
        [
            [
                encoder.scalls[j](lx.iloc[i, j], rx.iloc[i, j])
                for j in range(encoder.no_scalls)
            ]
            for i in range(lx.shape[0])
        ]
    )


def reference_label(matches, lp, rp):
    """The previous DataGenerator labelling: compare the pair with every match."""
    return float(any((matches.left == lp) & (matches.right == rp)))


def all_pairs(left, right):
    lpos = np.repeat(np.arange(len(left)), len(right))
    rpos = np.tile(np.arange(len(right)), len(left))
    return left.iloc[lpos], right.iloc[rpos]


def test_encoding_matches_reference_on_mixed_columns():
    # Strings, numbers, and a column used by several similarities
    lrows, rrows = all_pairs(left.iloc[:12], right.iloc[:12])

    np.testing.assert_array_equal(
        sencoder.encode_as_matrix(lrows, rrows),
        reference_encoding(sencoder, lrows, rrows),
    )


def test_encoding_matches_reference_on_extension_dtypes():
    frame = pd.DataFrame(
        {
            "name": pd.array(["metal gear", "metal slug", "gear"], dtype="string"),
            "year": pd.array([1998, 2003, 2015], dtype="Int64"),
        }
    )
    encoder = SimilarityEncoder(
        SimilarityMap({"name": ["levenshtein", "jaro"], "year": ["euclidean"]})
    )
    lrows, rrows = all_pairs(frame, frame)

    np.testing.assert_array_equal(
        encoder.encode_as_matrix(lrows, rrows),
        reference_encoding(encoder, lrows, rrows),
    )


def test_labels_match_reference_on_every_pair():
    gen = DataGenerator(
        smap, left, right, matches, batch_size=64, mismatch_share=1.0, shuffle=False
    )
    labels = np.concatenate([gen[i][1] for i in range(len(gen))])

    expected = [
        reference_label(matches, k // len(right), k % len(right)) for k in gen.indices
    ]
    np.testing.assert_array_equal(labels, np.array(expected, dtype=np.float32))
    assert labels.sum() == len(matches)
