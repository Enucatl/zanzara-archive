"""Command-line entry point for the Zanzara Archive application."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from zanzara_archive import __version__
from zanzara_archive.corpus import CorpusValidationError, load_manifest, verify_corpus, write_report


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without starting services or reading archive data."""
    parser = argparse.ArgumentParser(
        prog="zanzara",
        description="Local-first tools for the Zanzara audio archive.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    corpus = commands.add_parser("corpus", help="validate frozen archive sources")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    verify = corpus_commands.add_parser("verify", help="verify manifest hashes and media metadata")
    verify.add_argument("--manifest", required=True, help="path to the frozen corpus manifest")
    verify.add_argument("--archive-root", required=True, help="read-only archive root")
    verify.add_argument("--output", help="write the successful JSON report outside the archive")
    verify.add_argument(
        "--ffprobe", default="ffprobe", help="ffprobe executable (default: ffprobe)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the currently available application commands."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "corpus":
        parser.print_help()
        return 0
    try:
        manifest = load_manifest(arguments.manifest)
        report = verify_corpus(
            manifest,
            arguments.archive_root,
            ffprobe_binary=arguments.ffprobe,
        )
        if arguments.output and report["valid"]:
            write_report(report, arguments.output)
    except CorpusValidationError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["valid"] else 1
