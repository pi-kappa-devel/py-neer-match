from . import left, matches, right, smap
from neer_match.reasoning import RefutationModel
import pytest
import tensorflow as tf


def test_refutation():
    """Test refutation model."""
    model = RefutationModel(smap)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    model.fit(left, right, matches, refutation="title", epochs=1)
    assert model is not None, "Failed to fit refutation model with association."
    model.fit(left, right, matches, refutation={"title": "jaro_winkler"}, epochs=1)
    assert (
        model is not None
    ), "Failed to fit refutation model with association and similarity."
    model = RefutationModel(smap)
    model.compile()
    with pytest.raises(ValueError):
        model.fit(left, right, matches, refutation="no-title", epochs=1)
    with pytest.raises(ValueError):
        model.fit(left, right, matches, refutation={"title": "no-similarity"}, epochs=1)
    with pytest.raises(ValueError):
        model.fit(left, right, matches, refutation={"title": []}, epochs=1)
    with pytest.raises(ValueError):
        model.fit(left, right, matches, refutation={"title"}, epochs=1)
