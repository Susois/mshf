"""Backward-compatible shim: python -m mshf.train vẫn hoạt động."""
from mshf.cli.train import *  # noqa: F401, F403
from mshf.cli.train import main
import sys

if __name__ == "__main__":
    sys.exit(main())
