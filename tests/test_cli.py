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


def test_missing_weights_reports_cleanly_not_as_a_traceback(tmp_path, capsys):
    """A fresh clone has no weights. That is an expected state, not a crash."""
    import numpy as np
    from PIL import Image

    src = tmp_path / "in.png"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(src)

    code = cli.main([str(src), str(tmp_path / "out.png")])
    err = capsys.readouterr().err

    assert code == 1
    assert "export_onnx.py" in err, "the error should name the fix"
    assert "Traceback" not in err
