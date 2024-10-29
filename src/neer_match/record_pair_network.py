"""
Record pair network module.

This module contains functionality for instantiating, training, and using a record
matching network.
"""

from neer_match.field_pair_network import FieldPairNetwork
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf


class RecordPairNetwork(tf.keras.Model):
    """Record network class.

    The class creates networks for matching records from two datasets.
    """

    def __init__(
        self,
        similarity_map,
        initial_feature_width_scales=10,
        feature_depths=2,
        initial_record_width_scale=10,
        record_depth=4,
        **kwargs,
    ):
        """Initialize a record network object."""
        if not isinstance(similarity_map, SimilarityMap):
            raise ValueError(
                "Input similarity_map must be an instance of SimilarityMap."
            )
        if (
            not isinstance(initial_record_width_scale, int)
            or initial_record_width_scale < 1
        ):
            raise ValueError(
                "Input initial_record_width_scale must be a positive integer."
            )
        if not isinstance(record_depth, int) or record_depth < 1:
            raise ValueError("Input record_depth must be a positive integer.")
        # The remaining arguments are check in FieldPairNetwork.

        self.similarity_map = similarity_map
        self.initial_feature_width_scales = initial_feature_width_scales
        self.feature_depths = feature_depths
        self.initial_record_width_scale = initial_record_width_scale
        self.record_depth = record_depth

        no_assoc = similarity_map.no_associations()
        if isinstance(initial_feature_width_scales, int):
            initial_feature_width_scales = [initial_feature_width_scales] * no_assoc
        if isinstance(feature_depths, int):
            feature_depths = [feature_depths] * no_assoc

        super(RecordPairNetwork, self).__init__(**kwargs)

        self.field_networks = []
        for i, name in enumerate(similarity_map.association_names()):
            self.field_networks.append(
                FieldPairNetwork(
                    size=similarity_map.association_sizes()[i],
                    initial_width_scale=initial_feature_width_scales[i],
                    depth=feature_depths[i],
                    name=name,
                )
            )
        self.concat = tf.keras.layers.Concatenate()
        self.record_layers = []
        for i in range(record_depth):
            size = (initial_record_width_scale * no_assoc) // (i + 1)
            self.record_layers += [
                tf.keras.layers.Dense(
                    max(size, 2),
                    activation=tf.keras.activations.relu,
                    name=f"hidden_record_mixing_{i}",
                )
            ]
        self.record_layers += [
            tf.keras.layers.Dense(
                1,
                tf.keras.activations.sigmoid,
                name="record_classifier",
            )
        ]

    def get_config(self):
        """Return the configuration of the network."""
        config = super().get_config().copy()
        config.update(
            {
                "similarity_map": self.similarity_map,
                "initial_feature_width_scales": self.initial_feature_width_scales,
                "feature_depths": self.feature_depths,
                "initial_record_width_scale": self.initial_record_width_scale,
                "record_depth": self.record_depth,
            }
        )
        return config

    def call(self, inputs):
        """Run the network on input."""
        outputs = []
        for i, layer in enumerate(self.field_networks):
            outputs.append(layer(inputs[i]))
        output = self.concat(outputs)
        for layer in self.record_layers:
            output = layer(output)
        return output
