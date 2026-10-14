from . import left, matches, right, smap
from neer_match.data_generator import DataGenerator
from neer_match.matching_model import DLMatchingModel


def fitted(epochs=2, **kwargs):
    model = DLMatchingModel(smap)
    model.compile()
    history = model.fit(
        left, right, matches, epochs=epochs, batch_size=16, verbose=0, **kwargs
    )
    return model, history


def generator():
    return DataGenerator(
        smap, left, right, matches, batch_size=16, mismatch_share=1.0, shuffle=False
    )


def test_validation_frames_are_scored_each_epoch():
    _, history = fitted(validation_data=(left, right, matches))

    for key in ("val_loss", "val_f1", "val_mcc"):
        assert len(history.history[key]) == 2


def test_validation_generator_is_scored_each_epoch():
    _, history = fitted(validation_data=generator())

    assert len(history.history["val_f1"]) == 2


def test_evaluate_on_frames_is_unchanged():
    model, _ = fitted(epochs=1)

    result = model.evaluate(left, right, matches)

    assert {"TP", "FP", "TN", "FN", "Loss", "F1", "MCC"} <= set(result)


def test_evaluate_on_a_generator_defers_to_keras():
    model, _ = fitted(epochs=1)

    result = model.evaluate(x=generator(), return_dict=True, verbose=0)

    assert {"loss", "f1", "mcc"} <= set(result)
