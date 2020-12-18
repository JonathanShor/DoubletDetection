from .doubletdetection import BoostClassifier
from . import plot


# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "doubletdetection"
__version__ = importlib_metadata.version(package_name)

__all__ = ["BoostClassifier", "plot"]
