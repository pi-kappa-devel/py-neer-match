"""
Global explainability module.

This module provides customized global explainability functionality for entity
matching tasks.
"""

from neer_match.data_generator import DataGenerator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def partial_dependence_function(model, left, right, xfeatures):
    """Calculate the partial dependence of the model on the given keys."""
    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
    )

    offsets = model.similarity_map.association_offsets()
    names = model.similarity_map.association_names()

    xkeys = list(xfeatures.keys())
    positions = [model.similarity_map.keys().index(key) for key in xkeys]
    positions = {
        (y := len([x for x in offsets if x <= pos]) - 1): (pos - offsets[y])
        for pos in positions
    }

    xvalues = list(xfeatures.values())
    result = 0.0
    for features in generator:
        for i, (apos, spos) in enumerate(positions.items()):
            features[names[apos]][:, spos] = xvalues[i]
        preds = model(features)
        result += tf.reduce_sum(preds)
    result /= generator.no_pairs()
    return result.numpy()


def partial_dependence(model, left, right, key, n=50):
    """Calculate the partial dependence of a key over a domain grid."""
    return np.array(
        [
            partial_dependence_function(model, left, right, {key: i / n})
            for i in range(n + 1)
        ]
    )


def partial_dependence_feature_importance(model, left, right, key, n=50):
    """Calculate the feature importance of a key using partial dependence."""
    return np.std(partial_dependence(model, left, right, key, n))


def partial_dependence_plot(model, left, right, key, n=50):
    """Plot the partial dependence of a key."""
    domain = [i / n for i in range(n + 1)]
    values = partial_dependence(model, left, right, key, n)
    std = np.std(values)
    values_up = values + std
    values_down = values - std
    fig, ax = plt.subplots()
    ax.plot(domain, values, label="PD", color="blue", linestyle="-")
    ax.plot(domain, values_up, label="PD $\\pm$ std", color="blue", linestyle="--")
    ax.plot(domain, values_down, color="blue", linestyle="--")
    ax.legend()
    ax.set_xlabel(key)
    ax.set_ylabel("Probability")
    return fig


def accumulated_local_effect(model, left, right, xkey, xvalue, centered=True, k=50):
    """Calculate the accumulated local effect of a key over a domain grid."""
    if xvalue < 0 or xvalue > 1:
        raise ValueError("xvalue must be in the range [0, 1]")

    generator = DataGenerator(
        model.record_pair_network.similarity_map,
        left,
        right,
    )

    offsets = model.similarity_map.association_offsets()
    names = model.similarity_map.association_names()

    pos = model.similarity_map.keys().index(xkey)
    apos = len([x for x in offsets if x <= pos]) - 1
    spos = pos - offsets[apos]

    result = 0.0
    for i in range(k):
        term = 0.0
        count = 0
        for features in generator:
            x = features[names[apos]][:, spos]
            lvalue = xvalue * i / (k + 1)
            hvalue = xvalue * (i + 1) / (k + 1)
            xi = (x >= lvalue) & (x <= hvalue)
            used = {k: f[xi, :] for k, f in features.items()}
            used[names[apos]][:, spos] = lvalue
            lpreds = model(used)
            used[names[apos]][:, spos] = hvalue
            hpreds = model(used)
            term += tf.reduce_sum(hpreds - lpreds).numpy()
            count += sum(x)
        result += term / count
    return result


def accumulated_local_effect_plot(model, left, right, key, n=50, k=50):
    """Plot the accumulated local effect of a key."""
    domain = [i / n for i in range(n + 1)]
    values = [
        accumulated_local_effect(model, left, right, key, i, k=k) for i in domain
    ]
    std = np.std(values)
    values_up = values + std
    values_down = values - std
    fig, ax = plt.subplots()
    ax.plot(domain, values, label="ALE", color="blue", linestyle="-")
    ax.plot(domain, values_up, label="ALE $\\pm$ std", color="blue", linestyle="--")
    ax.plot(domain, values_down, color="blue", linestyle="--")
    ax.legend()
    ax.set_xlabel(key)
    ax.set_ylabel("$\\Delta$ Probability")
    return fig
