from . import smap
from neer_match.reasoning import RefutationModel
import pytest


def test_initialization():
    """Test refutation model initialization."""
    model = RefutationModel(smap, "title")
    assert model is not None, "Failed to initialize refutation model from association."
    model = RefutationModel(smap, "title", "jaro_winkler")
    assert (
        model is not None
    ), "Failed to initialize refutation model from association and similarity."
    with pytest.raises(ValueError):
        model = RefutationModel(smap, "no-title")
    with pytest.raises(ValueError):
        model = RefutationModel(smap, "title", "no-similarity")
