from .routes_dataset import router as dataset_router
from .routes_eda import router as eda_router
from .routes_stats import router as stats_router
from .routes_features import router as features_router
from .routes_models import router as models_router
from .routes_predict import router as predict_router

__all__ = [
    "dataset_router",
    "eda_router",
    "stats_router",
    "features_router",
    "models_router",
    "predict_router"
]
