"""
Deep learning matching models module.

This module contains functionality for instantiating, training, and evaluating deep
 learning matching models
"""

from neer_match.data_generator import DataGenerator
from neer_match.matching_model_tools import _evaluate_loop, _suggest
from neer_match.metrics import (
    PrecisionMetric,
    RecallMetric,
    AccuracyMetric,
    F1Metric,
    MCCMetric,
)
from neer_match.record_pair_network import RecordPairNetwork
from neer_match.similarity_map import SimilarityMap

import pandas as pd
import tensorflow as tf
import typing


class DLMatchingModel(tf.keras.Model):
    """A deep learning matching model class.

    Inherits :class:`tensorflow.keras.Model` and automates deep-learning-based entity
    matching using the similarity map supplied by the user.

    Attributes:
        record_pair_network (RecordPairNetwork): The record pair network.
    """

    def __init__(
        self,
        similarity_map: SimilarityMap,
        initial_feature_width_scales: typing.Union[int, typing.List[int]] = 10,
        feature_depths: typing.Union[int, typing.List[int]] = 2,
        initial_record_width_scale: int = 10,
        record_depth: int = 4,
        **kwargs,
    ) -> None:
        """Initialize a deep learning matching model.

        Generate a record pair network from the passed similarity map. The input
        arguments are passed to the record pair network (see
        :class:`.RecordPairNetwork`).

        Args:
            similarity_map: A similarity map object.
            initial_feature_width_scales: The initial width scales of the feature
                networks.
            feature_depths: The depths of the feature networks.
            initial_record_width_scale: The initial width scale of the record network.
            record_depth: The depth of the record network.
            **kwargs: Additional keyword arguments passed to parent class
                      (:class:`tensorflow.keras.Model`).
        """
        super().__init__(**kwargs)
        self.record_pair_network = RecordPairNetwork(
            similarity_map,
            initial_feature_width_scales=initial_feature_width_scales,
            feature_depths=feature_depths,
            initial_record_width_scale=initial_record_width_scale,
            record_depth=record_depth,
        )

    def build(self, input_shapes: typing.List[tf.TensorShape]) -> None:
        """Build the model."""
        super().build(input_shapes)
        self.record_pair_network.build(input_shapes)

    def call(self, inputs: typing.Dict[str, tf.Tensor]) -> tf.Tensor:
        """Call the model on inputs."""
        return self.record_pair_network(inputs)

    def compile(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        *,
        threshold: float = 0.5,
        **kwargs,
    ):
        """
        Compile the model with the desired loss, optimizer, and metrics.

        Args:
            optimizer: The optimizer to use.
            loss: The loss function to use.
            metrics: A list of metrics to compute during evaluation.
            threshold: Threshold for binary classification metrics (default 0.5).
            **kwargs: Additional arguments for tf.keras.Model.compile.
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        if loss is None:
            loss = tf.keras.losses.BinaryCrossentropy()
        self._threshold = float(threshold)
        if metrics is None:
            metrics = [
                AccuracyMetric(name="accuracy", threshold=threshold),
                PrecisionMetric(name="precision", threshold=threshold),
                RecallMetric(name="recall", threshold=threshold),
                F1Metric(name="f1", threshold=threshold),
                MCCMetric(name="mcc", threshold=threshold),
            ]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        batch_size: int = 16,
        mismatch_share: float = 0.1,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        """Fit the model.

        Construct a data generator from the input data frames using the
        similarity map with which the model was initialized and fit the model.
        The model is trained by calling the :func:`tensorflow.keras.Model.fit` method.

        Args:
            left: The left data frame.
            right: The right data frame.
            matches: The matches data frame.
            batch_size: Batch size.
            mismatch_share: Mismatch share.
            shuffle: Shuffle flag.
            **kwargs: Additional keyword arguments passed to parent class
                      (:func:`tensorflow.keras.Model.fit`).
        """
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            matches,
            batch_size=batch_size,
            mismatch_share=mismatch_share,
            shuffle=shuffle,
        )

        return super().fit(generator, **kwargs)

    def evaluate(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        matches: pd.DataFrame,
        batch_size: int = 16,
        mismatch_share: float = 1.0,
        **kwargs,
    ) -> dict:
        """
        Evaluate the DL model with the same return structure as the NS model's
        custom loop (no axioms). Returns TP, FP, TN, FN, Accuracy, Precision,
        Recall, F1, MCC, and the training loss summed across batches.
        """
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            matches,
            mismatch_share=mismatch_share,
            batch_size=batch_size,
            shuffle=False,
        )

        if not hasattr(self, "loss") or self.loss is None:
            raise ValueError(
                "Model must be compiled with a loss function before calling evaluate."
            )

        return _evaluate_loop(
            forward_fn=self.record_pair_network,
            generator=generator,
            base_loss_fn=self.loss,  # whatever you compiled with (incl. custom)
            threshold=getattr(self, "_threshold", 0.5),
            axioms=None,  # DL path → no axioms
        )

    def predict_from_generator(self, generator: DataGenerator, **kwargs) -> tf.Tensor:
        """Generate model predictions from a generator.

        Args:
            generator: The data generator.
            **kwargs: Additional keyword arguments passed to parent class
                      (:func:`tensorflow.keras.Model.predict`).
        """
        return super().predict(generator, **kwargs)

    def predict(
        self, left: pd.DataFrame, right: pd.DataFrame, batch_size: int = 16, **kwargs
    ) -> tf.Tensor:
        """Generate model predictions.

        Construct a data generator from the input data frames using the
        similarity map with which the model was initialized and generate predictions.

        Args:
            left: The left data frame.
            right: The right data frame.
            batch_size: Batch size.
            **kwargs: Additional keyword arguments passed to parent class
                      (:func:`tensorflow.keras.Model.predict`).
        """
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            batch_size=batch_size,
            mismatch_share=1.0,
            shuffle=False,
        )
        return self.predict_from_generator(generator, **kwargs)

    def suggest(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        count: int,
        batch_size: int = 16,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate model suggestions.

        Construct a data generator from the input data frames using the similarity map
        with which the model was initialized and generate suggestions.

        Args:
            left: The left data frame.
            right: The right data frame.
            count: The number of suggestions to generate.
            **kwargs: Additional keyword arguments passed to the suggest function.
        """
        return _suggest(self, left, right, count, batch_size=batch_size, **kwargs)

    @property
    def similarity_map(self) -> SimilarityMap:
        """Similarity Map of the Model."""
        return self.record_pair_network.similarity_map
