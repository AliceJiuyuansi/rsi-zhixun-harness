"""Platform entry point (copy of the official SDK template).

The platform runs this file as `describe`, `train` or `predict`; it only wires
the SDK CLI to our submission module (`contest_submission.py` by default).
"""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
