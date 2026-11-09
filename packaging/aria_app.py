"""What the packaged app runs when it is opened (see packaging/aria.spec)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.argv[1:] == ["--aria-check-bfloat16"]:
        # Aria trying bfloat16 in a separate process (aria.device.cpu_bf16_works).
        from aria.device import check_bf16_here
        check_bf16_here()
        sys.exit(0)
    from aria.app import main
    sys.exit(main())
