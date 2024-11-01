"""
Entity matching data generator module.

This module provides a data generation functionality for entity matching
tasks.
"""

from neer_match.similarity_encoding import SimilarityEncoder
import numpy as np
import pandas as pd
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator class.

    The class provides a data generator for entity matching tasks.
    """

    def __init__(
        self,
        similarity_map,
        left,
        right,
        matches=None,
        batch_size=32,
        mismatch_share=0.1,
        shuffle=False,
    ):
        """Initialize a data generator object."""
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

        self.dim = len(similarity_map)
        self.full_size = len(left) * len(right)
        self.used_size = self.full_size
        self.no_used_mismatches_per_match = None
        if self.matches is not None and self.mismatch_share < 1.0:
            no_matches = len(self.matches)
            no_total_mismatches = self.full_size - no_matches
            no_mismatches_per_match = max(no_total_mismatches // no_matches, 1)
            self.no_used_mismatches_per_match = min(
                int(self.mismatch_share * no_mismatches_per_match),
                no_mismatches_per_match,
            )
            self.used_size = self.no_used_mismatches_per_match * no_matches + no_matches

        self.batch_size = int(min(batch_size, self.used_size))
        self.no_batches = int(np.floor(self.used_size / self.batch_size))
        self.last_batch_size = self.used_size % self.batch_size
        if self.last_batch_size > 0:
            self.no_batches += 1

        self.similarity_map = similarity_map
        self.similarity_encoder = SimilarityEncoder(similarity_map)

        self.indices = self.__prepare_indices()

        self.on_epoch_end()

    def __prepare_indices(self):
        if self.no_used_mismatches_per_match is None:
            return np.arange(self.used_size, dtype=np.int32)

        indices = np.array([], dtype=np.int32)
        for i, (li, ri) in enumerate(self.matches.itertuples(index=False)):
            indices = np.append(indices, li * len(self.right) + ri)
        assert len(indices) == len(self.matches)
        assert len(indices) < self.used_size
        for li in range(len(self.left)):
            for ri in range(len(self.right)):
                if any(
                    (self.matches.iloc[:, 0] == li) & (self.matches.iloc[:, 1] == ri)
                ):
                    continue
                indices = np.append(indices, li * len(self.right) + ri)
                if len(indices) == self.used_size:
                    break
            if len(indices) == self.used_size:
                break
        assert len(indices) == self.used_size

        return indices

    def no_pairs(self):
        """Return the number of record pairs."""
        return self.used_size

    def no_matches(self):
        """Return the number of matches."""
        return len(self.matches)

    def no_mismatches(self):
        """Return the number of mismatches."""
        return self.used_size - self.no_matches()

    def __side_indices(self, indices):
        lpos = np.array([k // len(self.right) for k in indices], dtype=np.int32)
        rpos = np.array([k % len(self.right) for k in indices], dtype=np.int32)
        return lpos, rpos

    def __labels(self, lpos, rpos):
        if self.matches is None:
            return None
        labels = np.zeros((len(lpos)), dtype=np.float32)
        for i, lp in enumerate(lpos):
            labels[i] = any((self.matches.left == lp) & (self.matches.right == rpos[i]))
        return labels

    def __select_features(self, value):
        if self.matches is None:
            return None
        lpos, rpos = self.__side_indices(self.indices)
        labels = self.__labels(lpos, rpos)
        lpos = lpos[labels == value]
        rpos = rpos[labels == value]
        features = self.similarity_encoder(self.left.iloc[lpos], self.right.iloc[rpos])
        return features

    def matching_features(self):
        """Return the features of the matching pairs."""
        return self.__select_features(1.0)

    def non_matching_features(self):
        """Return the features of the non-matching pairs."""
        return self.__select_features(0.0)

    def __len__(self):
        """Return the number of batches per epoch."""
        return self.no_batches

    def __getitem__(self, index):
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

    def on_epoch_end(self):
        """Shuffle indices at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __str__(self):
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
