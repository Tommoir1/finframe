from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .database import Database


def default_data_dir() -> Path:
    configured = os.getenv("FINFRAME_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / "FinFrame").resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinFrame MaxN video annotation desktop application")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="Database, model and training-data directory")
    parser.add_argument("--database", type=Path, help="Override the SQLite database path")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    data_dir = arguments.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = arguments.database.expanduser().resolve() if arguments.database else data_dir / "finframe.sqlite3"
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from .main_window import MainWindow
    except ImportError as exc:
        print("FinFrame desktop dependencies are missing. Run: python -m pip install -e .", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        return 2

    application = QApplication(sys.argv[:1] + (argv or []))
    application.setApplicationName("FinFrame")
    application.setOrganizationName("FinFrame")
    try:
        database = Database(database_path)
        window = MainWindow(database, data_dir)
        window.show()
        return application.exec()
    except Exception as exc:
        QMessageBox.critical(None, "FinFrame could not start", str(exc))
        return 1
