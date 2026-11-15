# RetaillySale Backend Package
import os

# Set thread limits before NumPy / SciPy / OpenBLAS initialize
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .main import app

__all__ = ["app"]
