"""Backward-compatible shim: python -m mshf.detect vẫn hoạt động."""
from mshf.cli.detect import *  # noqa: F401, F403
from mshf.cli.detect import main
import sys

if __name__ == "__main__":
    sys.exit(main())
