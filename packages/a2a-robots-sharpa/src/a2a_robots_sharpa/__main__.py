"""``python -m a2a_robots_sharpa [cache_dir]``: download the model and print its path."""

import sys

from .model import fetch_model

print(fetch_model(sys.argv[1] if len(sys.argv) > 1 else None))
