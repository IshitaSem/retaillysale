from fastapi import APIRouter, Query, HTTPException
from sklearn.model_selection import train_test_split
from ..services.dataset_service import DatasetService
from ..ml.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_FEATURE
from ..ml.feature_selection import FeatureSelectionService
from ..schemas.features import FeatureSelectionResponse

router = APIRouter(prefix="/api/feature-selection", tags=["Feature Selection"])

@router.get("", response_model=FeatureSelectionResponse, summary="Get feature selection results")
def get_feature_selection(
    k: int = Query(9, ge=3, le=15, description="Number of top features to select")
):
    """Returns SelectKBest, correlation ranking, and Random Forest feature importances on train split."""
    try:
        df = DatasetService.get_instance().get_cleaned_data()
        feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X = df[feature_cols].copy()
        y = df[TARGET_FEATURE].copy()

        # Prevent data leakage by fitting selection on 80% train split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42)

        results = FeatureSelectionService.run_selection(X_train, y_train, k=k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform feature selection: {str(e)}")
