"""Command-line entry point for the Zanzara Archive application."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from zanzara_archive import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without starting services or reading archive data."""
    parser = argparse.ArgumentParser(
        prog="zanzara",
        description="Local-first tools for the Zanzara audio archive.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the currently available application commands."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
