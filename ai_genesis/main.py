"""CLI entry point for AI Genesis."""

from __future__ import annotations

import argparse

from ai_genesis.gui.app import run_gui
from ai_genesis.memory.sqlite_memory import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Genesis local MVP")
    parser.add_argument("--init-db", action="store_true", help="Initialize SQLite memory and exit")
    parser.add_argument("--gui", action="store_true", help="Start graphical interface")
    args = parser.parse_args()
    if args.init_db:
        MemoryStore().initialize()
        print("SQLite memory initialized")
        return
    run_gui()


if __name__ == "__main__":
    main()
