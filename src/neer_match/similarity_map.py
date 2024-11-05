"""
Similarity mappings module.

The module provides functionality to store and manage a similarity mappings
between records of two datasets.
"""

from rapidfuzz import distance
import numpy


def discrete(x, y):
    """Discrete similarity function."""
    return 1.0 if x == y else 0.0


def euclidean(x, y):
    """Euclidean similarity function."""
    return 1.0 / (1.0 + abs(x - y))


def gaussian(x, y):
    """Gaussian similarity function."""
    return numpy.exp(-((x - y) ** 2) / 2.0)


def available_similarities():
    """Return the list of available similarities."""
    return {
        "damerau_levenshtein": distance.DamerauLevenshtein.normalized_similarity,
        "discrete": discrete,
        "euclidean": euclidean,
        "gaussian": gaussian,
        "hamming": distance.Hamming.normalized_similarity,
        "indel": distance.Indel.normalized_similarity,
        "jaro": distance.Jaro.normalized_similarity,
        "jaro_winkler": distance.JaroWinkler.normalized_similarity,
        "lcsseq": distance.LCSseq.normalized_similarity,
        "levenshtein": distance.Levenshtein.normalized_similarity,
        "osa": distance.OSA.normalized_similarity,
        "postfix": distance.Postfix.normalized_similarity,
        "prefix": distance.Prefix.normalized_similarity,
    }


class SimilarityMap:
    """
    Similarity map class.

    The class stores a collection of associations between the records of two
    datasets.
    """

    def __init__(self, instructions):
        """Initialize a similarity map object."""
        if not isinstance(instructions, dict):
            raise ValueError("Input instructions must be a dictionary.")
        for key, value in instructions.items():
            if not isinstance(key, str):
                raise ValueError("Association key must be a string.")
            if not isinstance(value, list):
                raise ValueError(
                    "Association values must be a list. "
                    f"Instead got type {type(value)}."
                )
        self.instructions = instructions
        self.lcols = []
        self.rcols = []
        self.sims = []
        for association, similarities in self.instructions.items():
            parts = association.split("~")
            assert len(parts) == 1 or len(parts) == 2
            lcol = parts[0].strip()
            rcol = parts[1].strip() if len(parts) == 2 else lcol
            for similarity in similarities:
                self.lcols.append(lcol)
                self.rcols.append(rcol)
                self.sims.append(similarity)

    def no_associations(self):
        """Return the number of associations of the map."""
        return len(self.instructions)

    def association_sizes(self):
        """Return then number of similarities used by each association."""
        return [len(instruction) for instruction in self.instructions.values()]

    def association_offsets(self):
        """Return association offsets.

        Return the starting column offset of each association in the
        similarity matrix
        """
        return numpy.cumsum([0] + self.association_sizes()).tolist()[0:-1]

    def association_names(self):
        """Return a unique name for each association in the similarity map."""
        return [key.replace("~", "_") for key in self.instructions.keys()]

    def keys(self):
        """Return a unique key for each similarity map entry.

        Combine association with similarity names and return them.
        """
        return sum(
            [
                [f"{k.replace('~', '_')}_{s}" for s in v]
                for k, v in self.instructions.items()
            ],
            [],
        )

    def __iter__(self):
        """Iterate over the similarity map."""
        return zip(self.lcols, self.rcols, self.sims)

    def __len__(self):
        """Return the number of items in the similarity map."""
        return len(self.lcols)

    def __getitem__(self, index):
        """Return the item at the given index."""
        return self.lcols[index], self.rcols[index], self.sims[index]

    def __str__(self):
        """Return a string representation of the similarity map."""
        items = "\n  ".join([str(item) for item in self])
        return f"{self.__class__.__name__}[\n  {items}]"
