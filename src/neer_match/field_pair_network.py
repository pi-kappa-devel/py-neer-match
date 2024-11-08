"""
Field pair network module.

This module contains functionality for instantiating, training, and using a field pair
matching networks.
"""

import tensorflow as tf


class FieldPairNetwork(tf.keras.Model):
    """Field pair network class.

    The class creates networks for matching pairs of fields from two datasets.

    Attributes:
        size (int): The size of the input feature vectors.
        initial_width_scale (int): The initial width scale of the hidden layers.
        depth (int): The depth of the network.
    """

    def __init__(
        self, size: int, initial_width_scale: int = 10, depth: int = 2, **kwargs
    ) -> None:
        """Initialize a field pair network object.

        The network depth is determined from the depth parameter. The width of each
        hidden layer is determined by the initial width scale and the number of input
        features. The first hidden layer has a width of size * initial_width_scale, the
        second has a width of size * initial_width_scale / 2, and so on. The output
        layer has a sigmoid activation function.

        Args:
            size: The size of the input feature vectors.
            initial_width_scale: The initial width scale of the hidden layers.
            depth: The depth of the network.
            **kwargs: Additional keyword arguments passed to parent class
                      (tf.keras.Model).
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("Size must be a positive integer.")
        if not isinstance(initial_width_scale, int) or initial_width_scale < 1:
            raise ValueError("Initial width scale must be a positive integer.")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("Depth must be a positive integer.")

        self.size = size
        self.initial_width_scale = initial_width_scale
        self.depth = depth
        super().__init__(**kwargs)

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

    def get_config(self) -> dict:
        """Return the network configuration."""
        config = super().get_config().copy()
        config.update(
            {
                "size": self.size,
                "initial_width_scale": self.initial_width_scale,
                "depth": self.depth,
            }
        )
        return config

    def build(self, input_shape : tuple) -> None:
        """Build the network."""
        for layer in self.field_layers:
            layer.build(input_shape)
            input_shape = (input_shape[0], layer.units)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Run the network on inputs."""
        for layer in self.field_layers:
            inputs = layer(inputs)
        return inputs
