"""What the packaged app runs when it is opened (see packaging/aria.spec)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from aria.app import main
    sys.exit(main())
