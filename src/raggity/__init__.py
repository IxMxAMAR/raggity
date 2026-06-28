__version__ = "0.4.0"

# Register built-in connectors so resolve("connector", ...) works after
# `import raggity` without requiring callers to import sub-packages first.
from . import connectors as _connectors  # noqa: F401, E402
