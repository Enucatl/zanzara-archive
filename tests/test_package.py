"""CPU-only smoke tests for the installable application skeleton."""

import pytest

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


def test_web_requires_explicit_opt_in_for_non_loopback_bind(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["web", "--database", ".git/zanzara-state/test.db", "--host", "0.0.0.0"])

    assert error.value.code == 2
    assert "--allow-network" in capsys.readouterr().err
