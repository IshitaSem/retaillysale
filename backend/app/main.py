import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS
from .services.dataset_service import DatasetService
from .ml.trainer import ModelTrainer, ARTIFACTS_FILE
from .api import (
    dataset_router,
    eda_router,
    stats_router,
    features_router,
    models_router,
    predict_router
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retaillysale")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up dataset and train models if not already present
    logger.info("Initializing RetaillySale backend...")
    try:
        ds = DatasetService.get_instance()
        logger.info(f"Loaded dataset: {len(ds.get_cleaned_data())} clean records.")
        
        if not ARTIFACTS_FILE.exists():
            logger.info("Training baseline and nonlinear models...")
            ModelTrainer.train_and_save(k=6)
            logger.info("Models trained and artifacts saved.")
        else:
            logger.info("Found existing model artifacts.")
    except Exception as e:
        logger.error(f"Error during startup initialization: {e}")
    yield
    logger.info("Shutting down RetaillySale backend...")

app = FastAPI(
    title="RetaillySale API",
    description="Interactive Retail Sales Prediction & Feature Selection Platform (M.Sc. Data Science)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware - standard ASGI middleware with private network and regex origin support
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

# Register API Routers
app.include_router(dataset_router)
app.include_router(eda_router)
app.include_router(stats_router)
app.include_router(features_router)
app.include_router(models_router)
app.include_router(predict_router)

@app.get("/", tags=["System"])
def root():
    """Root endpoint returning API status, metadata, and available services."""
    return {
        "project": "RetaillySale",
        "description": "Retail Sales Prediction and Feature Selection API",
        "status": "online",
        "academic_subject": "M.Sc. Computer Science (Data Science)",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "dataset_summary": "/api/dataset/summary",
            "dataset_preview": "/api/dataset/preview",
            "eda": "/api/eda",
            "statistical_tests": "/api/statistical-tests",
            "feature_selection": "/api/feature-selection",
            "model_comparison": "/api/models/compare",
            "model_status": "/api/models/status",
            "predict": "/api/predict"
        }
    }

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for deployment monitoring (Render, Uptime, Docker)."""
    is_dataset_ready = False
    is_model_ready = False
    try:
        df = DatasetService.get_instance().get_cleaned_data()
        is_dataset_ready = len(df) > 0
    except Exception:
        pass

    is_model_ready = ARTIFACTS_FILE.exists()

    return {
        "status": "healthy" if (is_dataset_ready and is_model_ready) else "degraded",
        "dataset_ready": is_dataset_ready,
        "models_ready": is_model_ready
    }
