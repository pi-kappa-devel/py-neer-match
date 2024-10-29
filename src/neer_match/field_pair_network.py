"""
Field Pair Network.

This module contains functionality for instantiating, training, and using a field pair
matching networks.
"""

import tensorflow as tf


class FieldPairNetwork(tf.keras.Model):
    """Field Pair Network Class.

    The class creates networks for matching pairs of fields from two datasets.
    """

    def __init__(self, size, initial_width_scale=10, depth=2, **kwargs):
        """Initialize a field pair network object."""
        if not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer.")
        if not isinstance(initial_width_scale, int) or initial_width_scale < 1:
            raise ValueError("Initial width scale must be a positive integer.")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer.")

        self.size = size
        self.initial_width_scale = initial_width_scale
        self.depth = depth
        super(FieldPairNetwork, self).__init__(**kwargs)

        self.field_layers = []
        for i in range(depth):
            self.field_layers += [
                tf.keras.layers.Dense(
                    max(int(initial_width_scale * size / (i + 1)), 2),
                    activation=tf.keras.activations.relu,
                    name=f"{self.name.replace('~', '_')}_hidden_{i}",
                )
            ]
        self.field_layers += [
            tf.keras.layers.Dense(
                1,
                tf.keras.activations.sigmoid,
                name=f"{self.name}_classifier",
            )
        ]

    def get_config(self):
        """Return the configuration of the network."""
        config = super().get_config().copy()
        config.update(
            {
                "size": self.size,
                "initial_width_scale": self.initial_width_scale,
                "depth": self.depth,
            }
        )
        return config

    def call(self, inputs):
        """Run the network on input."""
        for layer in self.field_layers:
            inputs = layer(inputs)
        return inputs
