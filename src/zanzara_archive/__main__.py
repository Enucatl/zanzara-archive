"""Allow ``python -m zanzara_archive`` to invoke the application CLI."""

from zanzara_archive.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
