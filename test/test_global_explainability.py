from . import ns_model, left_short, right_short
from neer_match.global_explainability import (
    accumulated_local_effect_plot,
    partial_dependence,
    partial_dependence_feature_importance,
    partial_dependence_plot,
)
import numpy as np
import matplotlib


def test_partial_dependence():
    """Test partial dependence."""
    interp_n = 2
    result = partial_dependence(
        ns_model, left_short, right_short, "title_jaro_winkler", n=interp_n
    )
    assert isinstance(result, np.ndarray), (
        "Expected float result. " f"Instead got {type(result)}"
    )
    expected = (interp_n,)
    assert result.shape == expected, (
        f"Expected shape {expected}. " f"Instead got {result.shape}"
    )
    result = partial_dependence_feature_importance(
        ns_model, left_short, right_short, "title_jaro_winkler", n=interp_n
    )
    assert isinstance(result, np.float32), (
        "Expected float result. " f"Instead got {type(result)}"
    )
    assert result >= 0, "Expected positive result. " f"Instead got {result}"
    fig = partial_dependence_plot(
        ns_model, left_short, right_short, "title_jaro_winkler", n=interp_n
    )
    assert isinstance(
        fig, matplotlib.figure.Figure
    ), "Failed to generate partial dependence plot."


def test_accumulated_local_effect():
    """Test accumulated local effect."""
    simil_k = 2
    fig = accumulated_local_effect_plot(
        ns_model, left_short, right_short, "title_jaro_winkler", n=2, k=simil_k
    )
    assert isinstance(
        fig, matplotlib.figure.Figure
    ), "Failed to generate accumulated local effect plot."
