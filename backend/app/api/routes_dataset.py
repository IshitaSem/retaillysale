from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from ..services.dataset_service import DatasetService
from ..schemas.dataset import DatasetSummaryResponse, DatasetPreviewResponse

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

@router.get("/summary", response_model=DatasetSummaryResponse, summary="Get dataset summary and statistics")
def get_dataset_summary():
    """Returns dataset summary metrics, shapes, missing values, duplicates, and numeric/categorical statistics."""
    try:
        service = DatasetService.get_instance()
        return service.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset summary: {str(e)}")

@router.get("/preview", response_model=DatasetPreviewResponse, summary="Get paginated dataset preview")
def get_dataset_preview(
    dataset_type: str = Query("cleaned", description="Type of dataset: 'raw' or 'cleaned'"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Records per page"),
    search: Optional[str] = Query(None, description="Optional search term to filter records")
):
    """Returns paginated records from either the raw or cleaned dataset with optional search filter."""
    try:
        service = DatasetService.get_instance()
        return service.get_preview(
            dataset_type=dataset_type,
            page=page,
            page_size=page_size,
            search=search
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset preview: {str(e)}")
