"""deploy.py
========
Compatibility wrapper for the board-aware deploy script.

Use `deployment/deploy.py` going forward.
"""

from __future__ import annotations

from deployment.deploy import main


if __name__ == "__main__":
    main()