"""Enable ``python -m cardiorisk ...`` as an entry point."""

from __future__ import annotations

from cardiorisk.cli import app

if __name__ == "__main__":
    app()
