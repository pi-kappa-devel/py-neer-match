"""
Similarity encoding module.

The module provides functionality to store and manage a similarity encoders.
"""

from neer_match.similarity_map import available_similarities, SimilarityMap
import numpy
import pandas


class SimilarityEncoder:
    """Similarity encoder class.

    The class creates a similarity encoder from a similarity map. It can
    be used to encode pairs of records from two datasets.
    """

    def __init__(self, similarity_map):
        """Initialize a similarity encoder object."""
        if not isinstance(similarity_map, SimilarityMap):
            raise ValueError(
                "Input similarity_map must be an instance of SimilarityMap. "
                f"Instead got {type(similarity_map)}"
            )
        self.similarity_map = similarity_map
        self.scalls = []

        for i in range(len(self.similarity_map)):
            lcol, rcol, sim = self.similarity_map[i]
            scall = available_similarities()[sim]
            if scall is None:
                raise ValueError(f"Unknown similarity function: {sim}")
            self.scalls.append(scall)
        self.no_scalls = len(self.scalls)
        self.no_assoc = self.similarity_map.no_associations()
        self.assoc_begin = numpy.array(self.similarity_map.association_offsets())
        self.assoc_sizes = numpy.array(self.similarity_map.association_sizes())
        self.assoc_end = self.assoc_begin + self.assoc_sizes

    def __call__(self, left, right):
        """Encode one or more pair of records."""
        sim_matrix = self.encode_as_matrix(left, right)
        return [
            sim_matrix[:, self.assoc_begin[i] : self.assoc_end[i]]
            for i in range(self.no_assoc)
        ]

    def encode_as_matrix(self, left, right):
        """Encode a pair of records as a matrix."""
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"Left and right datasets must have the same number of records. "
                f"Instead got {left.shape[0]} (left) and {right.shape[0]} (right)."
            )

        lx = left[self.similarity_map.lcols]
        rx = right[self.similarity_map.rcols]

        if len(lx.shape) == 1:
            vector = numpy.array(
                [self.scalls[i](lx.iloc[i], rx.iloc[i]) for i in range(self.no_scalls)]
            )
            vector.shape = (1, vector.shape[0])
            return vector
        else:
            return numpy.array(
                [
                    [
                        self.scalls[j](lx.iloc[i, j], rx.iloc[i, j])
                        for j in range(self.no_scalls)
                    ]
                    for i in range(lx.shape[0])
                ]
            )

    def encoded_shape(self, batch_size=-1):
        """Return the shape of the encoded data."""
        return [(batch_size, sz) for sz in self.assoc_sizes]

    def report_encoding(self, left, right):
        """Report encoding of a pair of records."""
        smatrix = self.encode_as_matrix(left, right)
        report = []
        if isinstance(left, pandas.Series) and isinstance(right, pandas.Series):
            to = 1
        elif isinstance(left, pandas.DataFrame) and isinstance(right, pandas.DataFrame):
            to = left.shape[0]
        else:
            raise ValueError("Left and right must be a pandas Series or DataFrame.")
        for pos in range(to):
            if isinstance(left, pandas.Series):
                lseries = left[self.similarity_map.lcols]
                rseries = right[self.similarity_map.rcols]
            else:
                lseries = left.iloc[pos][self.similarity_map.lcols]
                rseries = right.iloc[pos][self.similarity_map.rcols]
            sseries = pandas.Series(smatrix[pos, :].tolist())
            lseries.name = "Left"
            lseries.index = self.similarity_map.keys()
            rseries.name = "Right"
            rseries.index = self.similarity_map.keys()
            sseries.name = "Similarities"
            sseries.index = self.similarity_map.keys()
            report.append(pandas.concat([lseries, rseries, sseries], axis=1))
        return report
