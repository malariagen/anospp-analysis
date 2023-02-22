from .data import data

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

# this will read version from pyproject.toml
__version__ = importlib_metadata.version(__name__)
