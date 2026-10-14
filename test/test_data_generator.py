from . import left_short, matches_short, right_short, smap
from neer_match.data_generator import DataGenerator
from neer_match.matching_model import NSMatchingModel
from neer_match.similarity_map import SimilarityMap
import numpy as np
import pandas as pd
import tensorflow as tf


def square_data(n=100):
    """n records per side; left record i matches right record i."""
    left = pd.DataFrame({"name": [f"name {i}" for i in range(n)]})
    right = pd.DataFrame({"name": [f"name {i}" for i in range(n)]})
    matches = pd.DataFrame({"left": range(n), "right": range(n)})
    return left, right, matches


def generator(mismatch_share=0.01, shuffle=True, n=100):
    left, right, matches = square_data(n)
    return DataGenerator(
        SimilarityMap({"name": ["levenshtein"]}),
        left,
        right,
        matches,
        batch_size=16,
        mismatch_share=mismatch_share,
        shuffle=shuffle,
    )


def split(indices, n=100):
    """Matching and mismatching pair indices, for the diagonal matches above."""
    indices = np.asarray(indices)
    is_match = indices // n == indices % n
    return indices[is_match], indices[~is_match]


def test_mismatches_are_drawn_from_the_whole_cross_product():
    np.random.seed(0)
    _, mismatches = split(generator().indices)

    # Drawn in row order, all 100 mismatches came from left records 0 and 1
    assert len(np.unique(mismatches // 100)) > 20


def test_sample_holds_every_match_and_distinct_mismatches():
    np.random.seed(0)
    gen = generator()
    matches, mismatches = split(gen.indices)

    assert len(gen.indices) == gen.used_size
    assert sorted(matches) == [i * 100 + i for i in range(100)]
    assert len(np.unique(mismatches)) == len(mismatches) == gen.no_mismatches()


def test_mismatches_are_redrawn_each_epoch_when_shuffling():
    np.random.seed(0)
    gen = generator(shuffle=True)
    before = set(split(gen.indices)[1])

    gen.on_epoch_end()

    matches, mismatches = split(gen.indices)
    assert set(mismatches) != before
    assert len(matches) == 100


def test_mismatches_stay_fixed_without_shuffling():
    np.random.seed(0)
    gen = generator(shuffle=False)
    before = gen.indices.copy()

    gen.on_epoch_end()

    np.testing.assert_array_equal(gen.indices, before)


def test_sample_follows_numpy_seed():
    np.random.seed(3)
    first = generator().indices
    np.random.seed(3)
    second = generator().indices

    np.testing.assert_array_equal(first, second)


def test_full_share_uses_every_pair():
    gen = generator(mismatch_share=1.0, n=10)

    assert sorted(gen.indices) == list(range(100))


def test_ns_training_starts_each_epoch_anew(monkeypatch):
    calls = []
    original = DataGenerator.on_epoch_end

    def counting_on_epoch_end(self):
        calls.append(self)
        original(self)

    monkeypatch.setattr(DataGenerator, "on_epoch_end", counting_on_epoch_end)
    model = NSMatchingModel(smap)
    # An optimizer of its own: compile's default instance is shared by every
    # NSMatchingModel, and building it here would break the other tests' model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    model.fit(
        left_short, right_short, matches_short, epochs=3, batch_size=10, verbose=0
    )

    assert len(calls) == 3
