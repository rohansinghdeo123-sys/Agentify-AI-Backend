"""Compatibility entry point for installing the verified Chapter 1 Master Notes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.index_amit_chapter1 import main


if __name__ == "__main__":
    main()
