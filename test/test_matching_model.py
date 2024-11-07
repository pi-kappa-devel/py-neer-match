from . import (
    dl_model,
    left_short,
    matches_short,
    ns_model,
    right_short,
    smap,
)
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
import pytest
import numpy as np
import pandas as pd


def test_initialization():
    """Test initialization of the model."""
    with pytest.raises(ValueError):
        DLMatchingModel("no-similarity-map")
    with pytest.raises(ValueError):
        NSMatchingModel(smap, initial_record_width_scale=-1)
    with pytest.raises(ValueError):
        DLMatchingModel(smap, record_depth=-1)


def test_compilation():
    """Test initialization of the model."""
    dl_model.compile(loss="binary_crossentropy")
    assert dl_model is not None, "Failed to compile the model."
    ns_model.compile()
    assert ns_model is not None, "Failed to compile the model."


def test_fit():
    """Test fitting the model."""
    dl_model.fit(left_short, right_short, matches_short, epochs=1, batch_size=10)
    assert dl_model is not None, "Failed to fit the model."
    ns_model.fit(left_short, right_short, matches_short, epochs=1, batch_size=10)
    assert ns_model is not None, "Failed to fit the model."
    with pytest.raises(ValueError):
        ns_model.fit(None, right_short, matches_short, epochs=1, batch_size=0)
    with pytest.raises(ValueError):
        ns_model.fit(left_short, None, matches_short, epochs=1, batch_size=0)
    with pytest.raises(ValueError):
        ns_model.fit(left_short, right_short, None, epochs=1, batch_size=0)
    with pytest.raises(ValueError):
        ns_model.fit(left_short, right_short, matches_short, epochs=1.0, batch_size=0)
    with pytest.raises(ValueError):
        ns_model.fit(left_short, right_short, matches_short, epochs=1, batch_size=-1)


def test_evaluate():
    """Test evaluating the model."""
    results = ns_model.evaluate(left_short, right_short, matches_short)
    assert isinstance(results, dict), "Failed to evaluate the model."
    results = dl_model.evaluate(left_short, right_short, matches_short)
    assert isinstance(results, float), "Failed to evaluate the model."


def test_predict():
    """Test predicting with the model."""
    dl_preds = dl_model.predict(left_short, right_short)
    assert isinstance(
        dl_preds, np.ndarray
    ), "Failed to predict with the deep-learning model."
    expected_shape = (len(left_short) * len(right_short), 1)
    assert dl_preds.shape == expected_shape, (
        "Unexpected prediction shape. "
        f"Expected: {expected_shape}, got: {dl_preds.shape}"
    )
    ns_preds = ns_model.predict(left_short, right_short)
    assert isinstance(
        ns_preds, np.ndarray
    ), "Failed to predict with the neural-symbolic model."
    assert ns_preds.shape == expected_shape, (
        "Unexpected prediction shape. "
        f"Expected: {expected_shape}, got: {ns_preds.shape}"
    )


def test_suggest():
    """Test suggesting with the model."""
    dl_suggestions = dl_model.suggest(left_short, right_short, count=3)
    assert isinstance(
        dl_suggestions, pd.DataFrame
    ), "Failed to suggest with the deep-learning model."
    expected_shape = (len(left_short) * 3, 3)
    assert dl_suggestions.shape == expected_shape, (
        "Unexpected suggestion shape. "
        f"Expected: {expected_shape}, got: {dl_suggestions.shape}"
    )
    ns_suggestions = ns_model.suggest(left_short, right_short, count=3)
    assert isinstance(
        ns_suggestions, pd.DataFrame
    ), "Failed to suggest with the neural-symbolic model."
    assert ns_suggestions.shape == expected_shape, (
        "Unexpected suggestion shape. "
        f"Expected: {expected_shape}, got: {ns_suggestions.shape}"
    )
