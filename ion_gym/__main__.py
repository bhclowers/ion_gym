"""`python -m ion_gym` -> the ion-gym CLI. The work runs only when the
module is executed, never on import (L-550)."""
import sys

from ion_gym.cli import main

if __name__ == "__main__":
    sys.exit(main())
