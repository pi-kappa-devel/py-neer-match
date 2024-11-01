"""
Local explainability module.

This module provides customized local explainability functionality for entity
matching tasks.
"""

from neer_match.data_generator import DataGenerator
import numpy as np
import pandas as pd
import tensorflow as tf


def lime(model, left, right, xindices, n=100):
    """Calculate local interpretable model-agnostic explanations."""
    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
    )
    xl = left.iloc[xindices[0]]
    xr = right.iloc[xindices[1]]
    x = generator.similarity_encoder.encode_as_matrix(xl, xr)
    sum_x = 0.0
    sum_xx = 0.0
    for features in generator:
        sum_x += tf.concat(
            [tf.reduce_sum(f, axis=0) for k, f in generator[0].items()], 0
        )
        sum_xx += tf.concat(
            [tf.reduce_sum(f**2, axis=0) for k, f in generator[0].items()], 0
        )
    mean = sum_x / generator.no_pairs()
    stddev = tf.sqrt(sum_xx / generator.no_pairs() - mean**2)
    sample = tf.random.normal((n, 6), mean=mean, stddev=stddev, dtype=tf.dtypes.float64)
    sample = tf.clip_by_value(sample, 0.0, 1.0)
    dist = sample - x
    weights = tf.exp(-tf.reduce_sum(dist**2, axis=1) / 2.0)

    begins = generator.similarity_encoder.assoc_begin
    ends = generator.similarity_encoder.assoc_end
    names = generator.similarity_map.association_names()
    features = {k: sample[:, begins[i] : ends[i]] for i, k in enumerate(names)}
    y = tf.cast(model(features), tf.dtypes.float64)

    weights = tf.linalg.diag(weights)
    X = tf.concat([tf.ones((n, 1), dtype=tf.dtypes.float64), sample], axis=1)
    u = tf.tensordot(X, weights, axes=[[0], [1]])
    p1 = tf.tensordot(u, X, axes=[[1], [0]])
    vcov = tf.linalg.inv(p1)
    p2 = tf.tensordot(u, y, axes=[[1], [0]])
    beta = tf.tensordot(vcov, p2, axes=[[1], [0]])

    return pd.DataFrame(
        {
            "feature": ["intercept"] + generator.similarity_map.keys(),
            "coef": beta.numpy().flatten(),
            "stderr": tf.sqrt(tf.linalg.diag_part(vcov)).numpy(),
        }
    )


def shap(model, left, right, xindices, xkey, iterations=100):
    """Calculate the Shapley value of a key."""
    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
    )

    offsets = np.array(model.similarity_map.association_offsets())
    names = model.similarity_map.association_names()
    n_positions = {}
    for i, key in enumerate(generator.similarity_map.keys()):
        if key == xkey:
            continue
        lower_than = offsets[offsets <= i]
        n_positions[i] = (len(lower_than) - 1, i - lower_than[-1])

    xl = left.iloc[xindices[0]]
    xr = right.iloc[xindices[1]]
    x_feature = generator.similarity_encoder(xl, xr)
    x_feature = {k: v for k, v in zip(names, x_feature)}
    x_aindex = generator.similarity_map.keys().index(xkey)
    xlower_than = offsets[offsets <= x_aindex]
    x_name = names[len(xlower_than) - 1]
    x_sindex = x_aindex - xlower_than[-1]

    n_count = len(n_positions)
    n_indices = list(n_positions.keys())
    shapley = 0.0
    for i in range(iterations):
        count = tf.random.uniform((), maxval=n_count, dtype=tf.dtypes.int32).numpy()
        indices = tf.random.shuffle(n_indices)[:count].numpy()
        positions = {pos: val for pos, val in n_positions.items() if pos in indices}
        zbatch = generator[
            tf.random.uniform((), maxval=len(generator), dtype=tf.dtypes.int32).numpy()
        ]
        zbindex = tf.random.uniform(
            (), maxval=generator.batch_size, dtype=tf.dtypes.int32
        ).numpy()
        z_feature = {k: v[zbindex].reshape(1, -1) for k, v in zbatch.items()}
        xp = {k: v.copy() for k, v in x_feature.items()}
        for pos, (apos, spos) in positions.items():
            name = names[apos]
            xp[name][:, spos] = z_feature[name][:, spos]
        xm = {k: v.copy() for k, v in xp.items()}
        xm[x_name][:, x_sindex] = z_feature[x_name][:, x_sindex]
        fp = model(xp)
        fm = model(xm)
        shapley += (fp - fm).numpy()[0, 0]
    shapley /= iterations

    return shapley
