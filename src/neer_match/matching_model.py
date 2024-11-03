"""
Matching models module.

This module contains functionality for instantiating, training, and evaluating
deep learning and neural-symbolic matching models
"""

from neer_match.axiom_generator import AxiomGenerator
from neer_match.data_generator import DataGenerator
from neer_match.record_pair_network import RecordPairNetwork
import pandas
import tensorflow as tf


def _suggest(model, left, right, count, batch_size=32, **kwargs):
    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
        mismatch_share=1.0,
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = model.predict_from_generator(generator, **kwargs)[
        : len(left) * len(right)
    ]
    sides = generator._DataGenerator__side_indices(generator.indices)
    features = pandas.DataFrame({"left": sides[0], "right": sides[1]})
    suggestions = features.assign(prediction=predictions)
    where = (
        suggestions.groupby((features.index / right.shape[0]).astype(int))["prediction"]
        .nlargest(count)
        .index.get_level_values(1)
    )
    return suggestions.iloc[where]


class DLMatchingModel(tf.keras.Model):
    """A deep learning matching model class.

    Inherits :func:`tensorflow.keras.Model` and automates deep learning based data
    matching. The matching problem is transformed to a classification problem based
    on a similarity map supplied by the user.
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
        """Initialize a deep learning matching model."""
        super().__init__(**kwargs)
        self.record_pair_network = RecordPairNetwork(
            similarity_map,
            initial_feature_width_scales=initial_feature_width_scales,
            feature_depths=feature_depths,
            initial_record_width_scale=initial_record_width_scale,
            record_depth=record_depth,
        )

    def build(self, input_shapes):
        """Build the model."""
        super().build(input_shapes)
        self.record_pair_network.build(input_shapes)

    def call(self, inputs):
        """Call the model on inputs."""
        return self.record_pair_network(inputs)

    def fit(self, left, right, matches, **kwargs):
        """Fit the model."""
        dg_kwargs = {}
        for key in ["batch_size", "mismatch_share", "shuffle"]:
            if key in kwargs:
                dg_kwargs[key] = kwargs.pop(key)
        generator = DataGenerator(
            self.record_pair_network.similarity_map, left, right, matches, **dg_kwargs
        )

        return super().fit(generator, **kwargs)

    def evaluate(self, left, right, matches, **kwargs):
        """Evaluate the model."""
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            matches,
            mismatch_share=1.0,
            shuffle=False,
        )
        return super().evaluate(generator, **kwargs)

    def predict_from_generator(self, generator, **kwargs):
        """Generate model predictions from a generator."""
        return super().predict(generator, **kwargs)

    def predict(self, left, right, **kwargs):
        """Generate model predictions."""
        gen_kwargs = {
            "mismatch_share": 1.0,
            "shuffle": False,
        }
        if "batch_size" in kwargs:
            gen_kwargs["batch_size"] = kwargs.pop("batch_size")
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            **gen_kwargs,
        )
        return self.predict_from_generator(generator, **kwargs)

    def suggest(self, left, right, count, **kwargs):
        """Generate model suggestions."""
        suggest_kwargs = {}
        if "batch_size" in kwargs:
            suggest_kwargs["batch_size"] = kwargs.pop("batch_size")
        return _suggest(self, left, right, count, **suggest_kwargs, **kwargs)

    @property
    def similarity_map(self):
        """Similarity Map of the Model."""
        return self.record_pair_network.similarity_map


class NSMatchingModel:
    """A neural-symbolic matching model class."""

    def __init__(
        self,
        similarity_map,
        initial_feature_width_scales=10,
        feature_depths=2,
        initial_record_width_scale=10,
        record_depth=4,
    ):
        """Initialize a neural-symbolic matching learning matching model."""
        self.record_pair_network = RecordPairNetwork(
            similarity_map,
            initial_feature_width_scales=initial_feature_width_scales,
            feature_depths=feature_depths,
            initial_record_width_scale=initial_record_width_scale,
            record_depth=record_depth,
        )

    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    ):
        """Compile the model."""
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = optimizer
        features = [
            tf.random.normal((1, size), dtype=tf.float32)
            for size in self.record_pair_network.similarity_map.association_sizes()
        ]
        self.record_pair_network(features)

    def __make_axioms(self, data_generator):
        axiom_generator = AxiomGenerator(data_generator)

        @tf.function
        def axioms():
            matching_axioms = axiom_generator.matching_axioms(self.record_pair_network)
            non_matching_axioms = axiom_generator.non_matching_axioms(
                self.record_pair_network
            )
            axioms = matching_axioms + non_matching_axioms
            kb = axiom_generator.FormAgg(axioms)
            sat = kb.tensor
            return sat

        return axioms

    def __for_epoch(
        self,
        data_generator,
        loss_clb,
        trainable_variables,
        verbose=1,
    ):
        no_batches = len(data_generator)
        pb_size = 60
        logs = {
            "no_batches": no_batches,
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
            "BCE": 0,
            "Sat": 0,
            "Loss": 0,
        }

        for i, (features, labels) in enumerate(data_generator):
            if verbose > 0:
                pb_step = int((i + 1) / no_batches * pb_size)
                pb = "=" * pb_step + "." * (pb_size - pb_step)
                print(f"\r[{pb}] {i + 1}/{no_batches}", end="", flush=True)
            with tf.GradientTape() as tape:
                batch_loss, batch_logs = loss_clb(features, labels)
            grads = tape.gradient(batch_loss, trainable_variables)
            self.optimizer.apply_gradients(zip(grads, trainable_variables))

            preds = batch_logs["Predicted"]
            preds = tf.reshape(preds, preds.shape[0])
            logs["TP"] += tf.reduce_sum(tf.round(preds) * labels)
            logs["FP"] += tf.reduce_sum(tf.round(preds) * (1.0 - labels))
            logs["TN"] += tf.reduce_sum((1.0 - tf.round(preds)) * (1.0 - labels))
            logs["FN"] += tf.reduce_sum((1.0 - tf.round(preds)) * labels)
            logs["BCE"] += batch_logs["BCE"]
            logs["Sat"] += batch_logs["Sat"]
            logs["Loss"] += batch_loss

        logs["Sat"] /= no_batches

        if verbose > 0:
            print("\r", end="", flush=True)

        return logs

    def __make_loss(self, axioms, satisfiability_weight):
        @tf.function
        def loss_clb(features, labels):
            preds = self.record_pair_network(features)
            bce = self.bce(labels, preds)
            sat = axioms()
            loss = (1.0 - satisfiability_weight) * bce + satisfiability_weight * (
                1.0 - sat
            )
            logs = {"BCE": bce, "Sat": sat, "Predicted": preds}
            return loss, logs

        return loss_clb

    def __traininig_loop(
        self, data_generator, loss_clb, trainable_variables, epochs, verbose, log_mod_n
    ):
        if verbose > 0:
            print(
                f"| {'Epoch':<10} | {'BCE':<10} | {'Rec':<10} | {'Prec':<10} "
                f"| {'F1':<10} | {'Sat':<10} |"
            )

        for epoch in range(epochs):
            logs = self.__for_epoch(
                data_generator,
                loss_clb,
                trainable_variables,
                verbose=verbose - 1,
            )
            recall = logs["TP"] / (logs["TP"] + logs["FN"])
            precision = logs["TP"] / (logs["TP"] + logs["FP"])
            f1 = 2.0 * precision * recall / (precision + recall)
            if verbose > 0 and epoch % log_mod_n == 0:
                print(
                    f"| {epoch:<10} | {logs['BCE'].numpy():<10.4f} | {recall:<10.4f} "
                    f"| {precision:<10.4f} | {f1:<10.4f} "
                    f"| {logs['Sat'].numpy():<10.4f} |"
                )
        if verbose > 0:
            print(
                f"Training finished at Epoch {epoch} with "
                f"DL loss {logs['BCE'].numpy():.4f} and "
                f"Sat {logs['Sat'].numpy():.4f}"
            )

    def fit(
        self,
        left,
        right,
        matches,
        epochs,
        satisfiability_weight=1.0,
        verbose=1,
        log_mod_n=1,
        **kwargs,
    ):
        """Fit the model."""
        if not isinstance(left, pandas.DataFrame):
            raise ValueError("Left must be a pandas DataFrame")
        if not isinstance(right, pandas.DataFrame):
            raise ValueError("Right must be a pandas DataFrame")
        if not isinstance(matches, pandas.DataFrame):
            raise ValueError("Matches must be a pandas DataFrame")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("Epochs must be an integer greater than 0")
        if satisfiability_weight < 0.0 or satisfiability_weight > 1.0:
            raise ValueError("Satisfiability weight must be between 0 and 1")
        if not isinstance(verbose, int):
            raise ValueError("Verbose must be an integer")
        if not isinstance(log_mod_n, int) or log_mod_n < 1:
            raise ValueError("Log mod n must be an integer greater than 0")

        data_generator = DataGenerator(
            self.record_pair_network.similarity_map, left, right, matches, **kwargs
        )

        axioms = self.__make_axioms(data_generator)
        loss_clb = self.__make_loss(axioms, satisfiability_weight)

        trainable_variables = self.record_pair_network.trainable_variables
        self.__traininig_loop(
            data_generator, loss_clb, trainable_variables, epochs, verbose, log_mod_n
        )

    def evaluate(self, left, right, matches, batch_size=32, satisfiability_weight=0.5):
        """Evaluate the model."""
        data_generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            matches,
            mismatch_share=1.0,
            batch_size=batch_size,
            shuffle=False,
        )

        axioms = self.__make_axioms(data_generator)
        loss_clb = self.__make_loss(axioms, satisfiability_weight)

        trainable_variables = self.record_pair_network.trainable_variables
        logs = self.__for_epoch(
            data_generator,
            loss_clb,
            trainable_variables,
            verbose=1,
        )

        tp = logs["TP"]
        fp = logs["FP"]
        tn = logs["TN"]
        fn = logs["FN"]
        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2.0 * precision * recall / (precision + recall)
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"Satisfiability: {logs['Sat'].numpy():.4f}")

    def predict_from_generator(self, generator):
        """Generate model predictions from a generator."""
        preds = self.record_pair_network(generator[0])
        for i, features in enumerate(generator):
            if i == 0:
                continue
            preds = tf.concat([preds, self.record_pair_network(features)], axis=0)
        return preds

    def predict(self, left, right, batch_size=32):
        """Generate model predictions."""
        generator = DataGenerator(
            self.record_pair_network.similarity_map,
            left,
            right,
            mismatch_share=1.0,
            batch_size=batch_size,
            shuffle=False,
        )
        return self.predict_from_generator(generator)

    def suggest(self, left, right, count, batch_size=32):
        """Generate model suggestions."""
        return _suggest(self, left, right, count, batch_size=batch_size)

    @property
    def similarity_map(self):
        """Similarity Map of the Model."""
        return self.record_pair_network.similarity_map
