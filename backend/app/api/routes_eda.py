from fastapi import APIRouter, HTTPException
from ..services.eda_service import EDAService
from ..schemas.eda import EDAResponse

router = APIRouter(prefix="/api/eda", tags=["Exploratory Data Analysis"])

@router.get("", response_model=EDAResponse, summary="Get EDA statistics, trends, and chart data")
def get_eda():
    """Returns exploratory data analysis metrics, trends over time, store mix, promo comparisons, and correlation matrix."""
    try:
        return EDAService.get_eda_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate EDA metrics: {str(e)}")
