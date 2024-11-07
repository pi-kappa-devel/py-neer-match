from . import dl_model, left_short, right_short
from neer_match.local_explainability import lime, shap
import pandas as pd


def test_lime():
    """Test LIME explainability"""
    results = lime(dl_model, left_short, right_short, [0, 1], n=8)
    assert isinstance(results, pd.DataFrame), "Failed to explain with LIME."
    no_expected_rows = len(dl_model.similarity_map) + 1
    assert (
        results.shape[0] == no_expected_rows
    ), f"Expected {len(left_short)} explanations, got {results.shape[0]}"


def test_shap():
    """Test SHAP explainability."""
    results = shap(
        dl_model,
        left_short,
        right_short,
        [0, 1],
        "title_jaro_winkler",
        iterations=1,
    )
    assert isinstance(results, float), "Failed to explain with SHAP."
