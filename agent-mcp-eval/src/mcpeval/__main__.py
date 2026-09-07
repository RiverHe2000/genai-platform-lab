"""Entry point for ``python -m mcpeval``.

The installed ``mcpeval`` console script and this module are the same command, which matters
because they are used in different places: the script in the Makefile and in
``scripts/run_experiments.sh``, this module wherever a host application must launch the MCP
server with a specific interpreter --- ``sys.executable -m mcpeval serve`` --- rather than
trusting whichever ``mcpeval`` happens to be first on the path.
"""

from __future__ import annotations

from mcpeval.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
