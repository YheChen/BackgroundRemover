import pytest

from bgremover import cli


def test_bare_paths_imply_the_single_image_command():
    args = cli.build_parser().parse_args(["i", "a.jpg", "b.png"])
    assert args.command == "i"
    assert args.edge == "matte"


def test_default_edge_mode_is_not_naive():
    """NAIVE is what makes free background removers look free."""
    args = cli.build_parser().parse_args(["i", "a.jpg", "b.png"])
    assert args.edge == "matte"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("#ffffff", (255, 255, 255)), ("fff", (255, 255, 255)), ("#0a7ea4", (10, 126, 164))],
)
def test_parse_colour(text, expected):
    assert cli._parse_colour(text) == expected


def test_parse_colour_rejects_nonsense():
    with pytest.raises(ValueError, match="cannot parse colour"):
        cli._parse_colour("chartreuse")


def test_models_command_runs(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "birefnet-general" in out
    assert "MIT" in out


def test_missing_weights_reports_cleanly_not_as_a_traceback(tmp_path, capsys, monkeypatch):
    """A fresh clone has no weights. That is an expected state, not a crash.

    Points BGREMOVER_WEIGHTS at an empty directory so the test holds whether
    or not the developer running it has already exported a graph. The ORT
    session is lru_cached, so that cache has to be cleared too.
    """
    import numpy as np
    from PIL import Image

    from bgremover.stages import segment

    monkeypatch.setenv("BGREMOVER_WEIGHTS", str(tmp_path / "empty-weights"))
    segment._onnx_session.cache_clear()

    src = tmp_path / "in.png"
    Image.fromarray(np.full((32, 32, 3), 127, np.uint8)).save(src)

    try:
        code = cli.main([str(src), str(tmp_path / "out.png")])
        err = capsys.readouterr().err
        assert code == 1
        assert "export_onnx.py" in err, f"error should name the fix, got: {err!r}"
        assert "Traceback" not in err
    finally:
        segment._onnx_session.cache_clear()


def test_no_subject_gives_a_clear_message_not_a_pymatting_internal(tmp_path, capsys):
    """A flat image has no salient object. The error should say that."""
    import numpy as np

    from bgremover.stages import matte, trimap
    from bgremover.types import TRIMAP_BG

    flat = np.zeros((64, 64, 3), np.uint8)
    empty = np.full((64, 64), TRIMAP_BG, np.uint8)
    with pytest.raises(ValueError, match="no foreground found"):
        matte.solve(flat, empty)
    assert trimap.band_fraction(empty) == 0.0
