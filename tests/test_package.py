"""CPU-only smoke tests for the installable application skeleton."""

from zanzara_archive import __version__
from zanzara_archive.cli import main


def test_package_imports_with_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_version(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0

    assert capsys.readouterr().out.strip() == __version__
