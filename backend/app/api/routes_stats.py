from fastapi import APIRouter, HTTPException
from ..services.stats_service import StatsService
from ..schemas.stats import StatsResponse

router = APIRouter(prefix="/api/statistical-tests", tags=["Statistical Testing"])

@router.get("", response_model=StatsResponse, summary="Get statistical tests and academic interpretations")
def get_statistical_tests():
    """Returns Mann-Whitney U test, Spearman rank correlation, Chi-Square test, and ANOVA results."""
    try:
        return StatsService.get_statistical_tests()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run statistical tests: {str(e)}")
