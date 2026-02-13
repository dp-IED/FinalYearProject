"""
Standalone comparison script with explicit result paths.
Generates rich HTML reports. Can be invoked by run.py compare or run directly.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evals.compare_methods import main

if __name__ == "__main__":
    main()
