"""
Entity matching data generator module.

This module provides a data generation functionality for entity matching tasks.
"""

from neer_match.similarity_map import SimilarityMap
from neer_match.similarity_encoding import SimilarityEncoder
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
import typing


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator class.

    The class provides a data generator for entity matching tasks. It inherits
    from the tf.keras.utils.Sequence class. Instances generate
    batches of similarities for the associated fields of two records in the
    cross product of the left and right data frames. The cross product is not explicitly
    computed. Instead, instances emulate it using indexing calculations.

    Attributes:
        left (pandas.DataFrame): The left DataFrame.
        right (pandas.DataFrame): The right DataFrame.
        matches (pandas.DataFrame): The matches DataFrame.
        batch_size (int): Batch size.
        mismatch_share (float): Mismatches share.
        shuffle (bool): Shuffle flag.
        full_size (int): The number of potential the record pairs.
        used_size (int): The used size of the record pairs.
        no_used_mismatches_per_match (int): The number of used mismatches per match.
        no_batches (int): The number of batches per epoch.
        last_batch_size (int): The size of the last batch.
        similarity_map (SimilarityMap): The similarity map.
        similarity_encoder (SimilarityEncoder): The similarity encoder.
        indices (numpy.ndarray): The used indices for the record pairs.
    """

    def __init__(
        self,
        similarity_map: SimilarityMap,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame = None,
        batch_size: int = 32,
        mismatch_share: float = 0.1,
        shuffle: bool = False,
    ) -> None:
        """Initialize a data generator object.

        Prepare the indexing variables that are used in the data generation process.

        Args:
            similarity_map: A similarity map object.
            left: The left DataFrame.
            right: The right DataFrame.
            matches: The matches DataFrame.
            batch_size: Batch size.
            mismatch_share: Mismatches share. Below 1, the used mismatches are
                drawn uniformly at random from all mismatching pairs, and drawn
                anew at the end of each epoch if shuffle is set.
            shuffle: Shuffle flag.
        """
        if not isinstance(left, pd.DataFrame):
            raise ValueError("Input left must be a pandas DataFrame.")
        if not isinstance(right, pd.DataFrame):
            raise ValueError("Input right must be a pandas DataFrame.")
        if not isinstance(matches, pd.DataFrame) and matches is not None:
            raise ValueError("Input matches must be a pandas DataFrame or None.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Input batch_size must be a positive integer.")
        if (
            not isinstance(mismatch_share, float)
            or mismatch_share < 0.0
            or mismatch_share > 1.0
        ):
            raise ValueError("Input mismatch_share must be a float in [0, 1].")
        if not isinstance(shuffle, bool):
            raise ValueError("Input shuffle must be a boolean.")
        # The similarity_map is check in SimilarityEncoder and RecordPairNetwork.

        super().__init__()
        self.left = left
        self.right = right
        self.matches = matches

        self.mismatch_share = mismatch_share
        self.shuffle = shuffle

        self.full_size = len(left) * len(right)
        self.used_size = self.full_size
        self.no_used_mismatches_per_match = None
        if self.matches is not None and self.mismatch_share < 1.0:
            no_matches = len(self.matches)
            no_total_mismatches = self.full_size - no_matches
            no_mismatches_per_match = max(no_total_mismatches // no_matches, 1)
            self.no_used_mismatches_per_match = max(
                int(self.mismatch_share * no_mismatches_per_match),
                1,
            )
            self.used_size = self.no_used_mismatches_per_match * no_matches + no_matches
        self.batch_size = int(min(batch_size, self.used_size))
        self.no_batches = int(np.floor(self.used_size / self.batch_size))
        self.last_batch_size = self.used_size % self.batch_size
        if self.last_batch_size > 0:
            self.no_batches += 1
        else:
            self.last_batch_size = self.batch_size

        self.similarity_map = similarity_map
        self.similarity_encoder = SimilarityEncoder(similarity_map)

        self.indices = self.__prepare_indices()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __prepare_indices(self) -> np.ndarray:
        if self.no_used_mismatches_per_match is None:
            return np.arange(self.used_size, dtype=np.int32)

        lpos = self.matches.iloc[:, 0].to_numpy(dtype=np.int64)
        rpos = self.matches.iloc[:, 1].to_numpy(dtype=np.int64)
        match_indices = lpos * len(self.right) + rpos
        indices = np.concatenate(
            [match_indices, self.__sample_mismatches(match_indices)]
        )
        assert len(indices) == self.used_size

        return indices

    def __sample_mismatches(self, match_indices: np.ndarray) -> np.ndarray:
        """Draw the used mismatches uniformly at random from all mismatching pairs.

        Taking the first pairs in row order instead would draw every mismatch
        from the first few left records. Candidates are drawn without
        replacement, as many as needed plus the number of matches, so that
        enough remain after the matches among them are removed. The draw is
        seeded from numpy's global random state, so that numpy.random.seed and
        keras.utils.set_random_seed make it reproducible.
        """
        no_mismatches = self.used_size - len(match_indices)
        excluded = np.unique(match_indices)
        rng = np.random.default_rng(np.random.randint(2**31))
        candidates = rng.choice(
            self.full_size, size=no_mismatches + len(excluded), replace=False
        )
        return candidates[~np.isin(candidates, excluded)][:no_mismatches]

    def no_pairs(self) -> int:
        """Return the number of record pairs."""
        return self.used_size

    def no_matches(self) -> int:
        """Return the number of matches."""
        return len(self.matches)

    def no_mismatches(self) -> int:
        """Return the number of mismatches."""
        return self.used_size - self.no_matches()

    def __side_indices(
        self, indices: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        lpos = np.array([k // len(self.right) for k in indices], dtype=np.int32)
        rpos = np.array([k % len(self.right) for k in indices], dtype=np.int32)
        return lpos, rpos

    @functools.cached_property
    def _match_pairs(self) -> typing.Set[typing.Tuple]:
        """The (left, right) pairs in matches, for constant-time lookups."""
        return set(zip(self.matches.left.tolist(), self.matches.right.tolist()))

    def __labels(self, lpos: np.ndarray, rpos: np.ndarray) -> np.ndarray:
        if self.matches is None:
            return None
        # A set lookup per pair, instead of comparing each pair with every match
        return np.array(
            [pair in self._match_pairs for pair in zip(lpos.tolist(), rpos.tolist())],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.no_batches

    def __getitem__(
        self, index: int
    ) -> typing.Union[dict, typing.Tuple[dict, np.ndarray]]:
        """Get the batch at the given index."""
        begin = index * self.batch_size
        if index < self.no_batches - 1:
            end = begin + self.batch_size
        elif index == self.no_batches - 1:
            end = begin + self.last_batch_size
        else:
            raise IndexError("Invalid batch index")
        indices = self.indices[begin:end]
        lpos, rpos = self.__side_indices(indices)

        features = self.similarity_encoder(self.left.iloc[lpos], self.right.iloc[rpos])
        features = {
            key: features[i]
            for i, key in enumerate(self.similarity_map.association_names())
        }

        if self.matches is None:
            return features

        labels = self.__labels(lpos, rpos)
        return features, labels

    def on_epoch_end(self) -> None:
        """Maybe shuffle indices at the end of each epoch.

        With shuffling enabled and only a share of the mismatches in use, the
        used mismatches are also drawn anew, so that every epoch trains on a
        different sample of them.
        """
        if self.shuffle:
            if self.no_used_mismatches_per_match is not None:
                self.indices = self.__prepare_indices()
            np.random.shuffle(self.indices)

    def __str__(self) -> str:
        """Return a string representation of the data generator."""
        items = "\n  ".join(
            [
                self.similarity_map.__str__().replace("\n", "\n  "),
                f"No Batches[{self.no_batches}]",
                f"Batch size[{self.batch_size}]",
                f"No Pairs[{self.no_pairs()}]",
                f"No Matches[{self.no_matches()}]",
                f"No Mismatches[{self.no_mismatches()}]",
            ]
        )
        return f"{self.__class__.__name__}[\n  {items}]"
