import pytest

from bgremover import models


def test_default_is_registered_and_permissive():
    spec = models.get(models.DEFAULT_MODEL)
    assert spec.licence in models.PERMISSIVE_LICENCES


def test_every_registered_model_is_permissively_licensed():
    for key in models.available():
        assert models.get(key).licence in models.PERMISSIVE_LICENCES, key


def test_non_commercial_weights_are_rejected():
    """The guard that keeps BRIA RMBG (CC BY-NC 4.0) out of the repo."""
    with pytest.raises(models.LicenceError, match="allow-list"):
        models.ModelSpec(
            key="rmbg-2.0",
            licence="CC-BY-NC-4.0",
            hf_repo="briaai/RMBG-2.0",
            onnx_filename="rmbg2.onnx",
            input_size=1024,
        )


def test_unknown_model_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        models.get("does-not-exist")
