import os
from pathlib import Path

# Configure threading limits to prevent OpenBLAS exhaustion on Windows Python 3.14
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dotenv import load_dotenv

# Load .env file from backend directory if present
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

PROJECT_ROOT = BACKEND_DIR.parent

def find_file(relative_subpath: str) -> Path:
    """Search common candidate locations for a file path."""
    candidates = [
        BACKEND_DIR / relative_subpath,
        PROJECT_ROOT / relative_subpath,
        PROJECT_ROOT / "data" / "raw" / "INDIA_RETAIL_DATA.xlsx",
        PROJECT_ROOT / "data" / "processed" / "retail_cleaned.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[1]

# Dataset paths
raw_env = os.getenv("DATA_RAW_PATH", "data/raw/INDIA_RETAIL_DATA.xlsx")
RAW_DATA_PATH = BACKEND_DIR / raw_env if not Path(raw_env).is_absolute() else Path(raw_env)
if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "INDIA_RETAIL_DATA.xlsx"

processed_env = os.getenv("DATA_PROCESSED_PATH", "data/processed/retail_cleaned.csv")
PROCESSED_DATA_PATH = BACKEND_DIR / processed_env if not Path(processed_env).is_absolute() else Path(processed_env)
if not PROCESSED_DATA_PATH.exists():
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retail_cleaned.csv"

# Models directory
models_env = os.getenv("MODELS_DIR", "models")
MODELS_DIR = BACKEND_DIR / models_env if not Path(models_env).is_absolute() else Path(models_env)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Server & CORS
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8081"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

cors_origins_str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8443,http://127.0.0.1:8443,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8081,http://127.0.0.1:8081",
)
CORS_ORIGINS = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
