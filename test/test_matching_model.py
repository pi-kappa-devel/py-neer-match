from . import dl_model, left, matches, ns_model, right, smap
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
import pytest


def test_initialization():
    """Test initialization of the model."""
    global model
    with pytest.raises(ValueError):
        model = DLMatchingModel("no-similarity-map")
    with pytest.raises(ValueError):
        model = NSMatchingModel(smap, initial_record_width_scale=-1)
    with pytest.raises(ValueError):
        model = DLMatchingModel(smap, record_depth=-1)
    assert dl_model is not None, "Failed to initialize the model."


def test_compilation():
    """Test initialization of the model."""
    dl_model.compile(loss="binary_crossentropy")
    ns_model.compile()
    assert dl_model is not None, "Failed to compile the model."


def test_fit():
    """Test fitting the model."""
    dl_model.fit(left, right, matches, epochs=1, batch_size=10)
    assert dl_model is not None, "Failed to fit the model."
