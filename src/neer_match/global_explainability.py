"""
Global explainability module.

This module provides customized global explainability functionality for entity matching
tasks.
"""

from neer_match.data_generator import DataGenerator
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import typing


def partial_dependence_function(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    xfeatures: dict[str, float],
) -> float:
    """Calculate the partial dependence of the model on the given keys.

    Replaces the values of the given keys with the specified values and
    calculates the average prediction of the model.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        xfeatures: A dictionary of the features with values where the partial
            dependence is calculated.
    """
    if not isinstance(model, (DLMatchingModel, NSMatchingModel)):
        raise ValueError(
            "The model argument must be an instance of DLMatchingModel "
            "or NSMatchingModel"
        )
    if not isinstance(xfeatures, dict):
        raise ValueError("The input xfeatures must be a dictionary")
    for k, v in xfeatures.items():
        if not isinstance(k, str):
            raise ValueError("The keys of xfeatures must be strings")
        if k not in model.similarity_map.keys():
            raise ValueError(f"Key {k} is not in the similarity map")
        if not isinstance(v, float) or v < 0 or v > 1:
            raise ValueError(
                "The values of xfeatures must be floats in the range [0, 1]"
            )

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
        preds = model.record_pair_network(features)
        result += tf.reduce_sum(preds)
    result /= generator.no_pairs()
    return result.numpy()


def partial_dependence(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    n: int = 50,
) -> np.ndarray:
    """Calculate the partial dependence of a key over a domain grid.

    Creates a [0, 1] grid of n interpolation points and calculates the partial
    dependence of the model on the key at each point.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        key: The key for which to calculate the partial dependence.
        n: The number of interpolation points to use.
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("Interpolation points (n) must be an integer greater than 1")
    # The remaining arguments are validated in partial_dependence_function
    return np.array(
        [
            partial_dependence_function(model, left, right, {key: i / (n - 1)})
            for i in range(n)
        ]
    )


def partial_dependence_feature_importance(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    n: int = 50,
):
    """Calculate the feature importance of a key using partial dependence.

    Calculates the standard deviation of the partial dependence of the model on the key
    over a [0, 1] domain grid.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        key: The key for which to calculate the feature importance.
        n: The number of interpolation points to use.
    """
    # Input arguments are validated in partial_dependence
    return np.std(partial_dependence(model, left, right, key, n))


def partial_dependence_plot(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    n: int = 50,
):
    """Plot the partial dependence of a key.

    Plots the partial dependence of the model on the key over a [0, 1] domain grid.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        key: The key for which to calculate the partial dependence.
        n: The number of interpolation points
    """
    domain = [i / (n - 1) for i in range(n)]
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


def accumulated_local_effect(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    xkey: str,
    xvalue: float,
    centered: bool = True,
    k: int = 50,
):
    """Calculate the accumulated local effect of a key over a domain grid.

    Creates a 0 to xvalue grid of n interpolation points, calculates local
    differences in the model's predictions for each segment of the grid, and
    averages the differences. If centered is True, the partial dependence of the
    model on the key at xvalue is subtracted from the accumulated local effect.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        xkey: The key for which to calculate the accumulated local effect.
        xvalue: The value of at which the accumulated local effect is calculated.
        centered: Whether to center the accumulated local effect.
        k: The number of interpolation points
    """
    if not isinstance(model, (DLMatchingModel, NSMatchingModel)):
        raise ValueError(
            "The model argument must be an instance of DLMatchingModel "
            "or NSMatchingModel"
        )
    if not isinstance(xkey, str):
        raise ValueError("The xkey argument must be a string")
    if xkey not in model.similarity_map.keys():
        raise ValueError(f"Key {xkey} is not in the similarity map")
    if xvalue < 0 or xvalue > 1:
        raise ValueError("xvalue must be in the range [0, 1]")
    if not isinstance(centered, bool):
        raise ValueError("The centered argument must be a boolean")
    if not isinstance(k, int) or k < 1:
        raise ValueError("The k argument must be an integer greater than 0")
    # The dataset arguments are validated in partial_dependence_function

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
            lpreds = model.record_pair_network(used)
            used[names[apos]][:, spos] = hvalue
            hpreds = model.record_pair_network(used)
            term += tf.reduce_sum(hpreds - lpreds).numpy()
            count += sum(x)
        result += term / count
    if centered:
        result -= partial_dependence_function(model, left, right, {xkey: xvalue})
    return result


def accumulated_local_effect_plot(
    model: typing.Union[DLMatchingModel, NSMatchingModel],
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    centered: bool = True,
    n: int = 50,
    k: int = 50,
):
    """Plot the accumulated local effect of a key.

    Args:
        model: The matching model to use.
        left: The left DataFrame.
        right: The right DataFrame.
        key: The key for which to calculate the accumulated local effect.
        centered: Whether to center the accumulated local effect.
        n: The number of interpolation points for the figure.
        k: The number of interpolation points for the local effect.
    """
    if not isinstance(n, int) or n < 2:
        raise ValueError("The n argument must be an integer greater than 1")
    # The remaining arguments are validated in accumulated_local_effect
    domain = [i / n for i in range(n + 1)]
    values = [accumulated_local_effect(model, left, right, key, i, k=k) for i in domain]
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
