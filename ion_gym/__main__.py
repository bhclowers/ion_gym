"""`python -m ion_gym` -> the ion-gym CLI. The work runs only when the
module is executed, never on import."""
import sys

from ion_gym.cli import main

if __name__ == "__main__":
    sys.exit(main())
