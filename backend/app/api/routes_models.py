from fastapi import APIRouter, Query, HTTPException
from ..ml.trainer import ModelTrainer, ARTIFACTS_FILE
from ..schemas.models import ModelComparisonResponse, ModelStatusResponse

router = APIRouter(prefix="/api/models", tags=["Machine Learning Models"])

@router.get("/compare", response_model=ModelComparisonResponse, summary="Compare baseline and nonlinear models")
def compare_models(
    k: int = Query(9, ge=3, le=15, description="Number of selected features for reduced model experiments")
):
    """Compares Linear Regression and Random Forest Regressor across All Features and Selected Features."""
    try:
        return ModelTrainer.get_comparison(k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model comparisons: {str(e)}")

@router.get("/status", response_model=ModelStatusResponse, summary="Get model training and loading status")
def get_model_status():
    """Returns whether models are trained and saved, and list of available model variants."""
    try:
        is_trained = ARTIFACTS_FILE.exists()
        last_trained = None
        if is_trained:
            from datetime import datetime
            import os
            mtime = os.path.getmtime(ARTIFACTS_FILE)
            last_trained = datetime.fromtimestamp(mtime).isoformat()

        return {
            "is_trained": is_trained,
            "models_available": [
                "Linear Regression (All Features)",
                "Linear Regression (Selected Features)",
                "Random Forest Regressor (All Features)",
                "Random Forest Regressor (Selected Features)"
            ],
            "artifacts_path": str(ARTIFACTS_FILE),
            "last_trained": last_trained
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model status: {str(e)}")
