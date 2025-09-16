"""
Matching models tools module.

This module contains functionality commoly used by DL and NS matching models.
"""

from __future__ import annotations

from neer_match.data_generator import DataGenerator
from neer_match.metrics import (
    PrecisionMetric,
    RecallMetric,
    AccuracyMetric,
    F1Metric,
    MCCMetric,
)

import heapq
import pandas as pd
import tensorflow as tf
import typing

if typing.TYPE_CHECKING:
    from neer_match.dl_matching_model import DLMatchingModel
    from neer_match.ns_matching_model import NSMatchingModel


def _suggest(
    model: DLMatchingModel | NSMatchingModel,
    left: pd.DataFrame,
    right: pd.DataFrame,
    count: int,
    batch_size: int = 32,
    **kwargs,
) -> pd.DataFrame:
    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
        mismatch_share=1.0,
        batch_size=batch_size,
        shuffle=False,
    )
    top_suggestions = {i: [] for i in range(len(left))}

    for batch_idx, batch in enumerate(generator):
        batch_pred = model.predict_from_generator(batch, **kwargs)
        for idx, pred in enumerate(batch_pred):
            array_index = batch_idx * batch_size + idx
            left_idx = array_index // len(right)
            right_idx = array_index % len(right)
            heapq.heappush(top_suggestions[left_idx], (float(pred), right_idx))
            if len(top_suggestions[left_idx]) > count:
                heapq.heappop(top_suggestions[left_idx])

    rows, cols, preds = [], [], []
    for left_idx, heap in top_suggestions.items():
        for pred, right_idx in sorted(heap, reverse=True):
            rows.append(left_idx)
            cols.append(right_idx)
            preds.append(pred)

    suggestions = pd.DataFrame({"left": rows, "right": cols, "prediction": preds})
    return suggestions


def _evaluate_loop(
    forward_fn: typing.Callable[[dict], tf.Tensor],
    generator: DataGenerator,
    base_loss_fn: tf.keras.losses.Loss,
    threshold: float,
    axioms: typing.Optional[typing.Callable[[dict, tf.Tensor], tf.Tensor]] = None,
    satisfiability_weight: float = 1.0,
    verbose: int = 1,
) -> dict:
    """
    Shared evaluation loop (no gradient updates).
    Returns TP, FP, TN, FN (integers), total Loss, and rate metrics.
    """

    # rate metrics (accumulate totals internally — not printed during training)
    precision_m = PrecisionMetric(threshold=threshold)
    recall_m = RecallMetric(threshold=threshold)
    accuracy_m = AccuracyMetric(threshold=threshold)
    f1_m = F1Metric(threshold=threshold)
    mcc_m = MCCMetric(threshold=threshold)

    # integer confusion counts (only for evaluation)
    tp_total = tf.constant(0, dtype=tf.int64)
    fp_total = tf.constant(0, dtype=tf.int64)
    tn_total = tf.constant(0, dtype=tf.int64)
    fn_total = tf.constant(0, dtype=tf.int64)

    total_loss = tf.constant(0.0, dtype=tf.float32)
    sat_sum = tf.constant(0.0, dtype=tf.float32)
    sat_batches = tf.constant(0.0, dtype=tf.float32)

    no_batches = len(generator)
    pb_size = 60

    for i, (features, labels) in enumerate(generator):
        if verbose > 0:
            pb_step = int((i + 1) / no_batches * pb_size)
            pb = "=" * pb_step + "." * (pb_size - pb_step)
            print(f"\r[{pb}] {i + 1}/{no_batches}", end="", flush=True)

        preds = forward_fn(features)
        base_loss = base_loss_fn(labels, preds)

        if axioms is not None:
            sat = axioms(features, labels)  # scalar in [0,1]
            loss = (1.0 - satisfiability_weight) * base_loss + satisfiability_weight * (
                1.0 - sat
            )
            sat_sum += tf.cast(sat, tf.float32)
            sat_batches += 1.0
        else:
            loss = base_loss

        # accumulate TOTAL loss consistent with reduction
        reduction = getattr(base_loss_fn, "reduction", tf.keras.losses.Reduction.AUTO)
        batch_n = tf.cast(tf.size(tf.reshape(labels, (-1,))), tf.float32)
        if reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            total_loss += loss * batch_n
        elif reduction == tf.keras.losses.Reduction.SUM:
            total_loss += loss
        else:
            total_loss += tf.reduce_sum(loss)

        # ---- update integer confusion counts (evaluation only) ----
        y_true_f = tf.reshape(tf.cast(labels, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.cast(preds, tf.float32), [-1])
        y_hat = tf.cast(y_pred_f >= tf.cast(threshold, tf.float32), tf.int32)
        y_true_i = tf.cast(tf.round(y_true_f), tf.int32)

        tp = tf.math.count_nonzero(
            tf.logical_and(y_hat == 1, y_true_i == 1), dtype=tf.int64
        )
        fp = tf.math.count_nonzero(
            tf.logical_and(y_hat == 1, y_true_i == 0), dtype=tf.int64
        )
        tn = tf.math.count_nonzero(
            tf.logical_and(y_hat == 0, y_true_i == 0), dtype=tf.int64
        )
        fn = tf.math.count_nonzero(
            tf.logical_and(y_hat == 0, y_true_i == 1), dtype=tf.int64
        )

        tp_total += tp
        fp_total += fp
        tn_total += tn
        fn_total += fn

        # ---- update rate metrics (computed from totals internally) ----
        for m in (precision_m, recall_m, accuracy_m, f1_m, mcc_m):
            m.update_state(labels, preds)

    if verbose > 0:
        print()

    result = {
        "TP": int(tp_total.numpy()),
        "FP": int(fp_total.numpy()),
        "TN": int(tn_total.numpy()),
        "FN": int(fn_total.numpy()),
        "Loss": float(total_loss.numpy()),
        "Accuracy": float(accuracy_m.result().numpy()),
        "Recall": float(recall_m.result().numpy()),
        "Precision": float(precision_m.result().numpy()),
        "F1": float(f1_m.result().numpy()),
        "MCC": float(mcc_m.result().numpy()),
    }

    if sat_batches > 0:
        result["Sat"] = float((sat_sum / sat_batches).numpy())

    return result


def _matching_model_or_raise(
    model: typing.Union[DLMatchingModel, NSMatchingModel]
) -> None:
    if not isinstance(model, (DLMatchingModel, NSMatchingModel)):
        raise ValueError(
            "The model argument must be an instance of DLMatchingModel "
            "or NSMatchingModel"
        )
