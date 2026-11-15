from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ShapeInfo(BaseModel):
    rows: int
    columns: int

class OutliersInfo(BaseModel):
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float

class NumericalFeatureStat(BaseModel):
    feature: str
    mean: float
    median: float
    std: float
    variance: float
    min: float
    q25: float
    q75: float
    max: float
    iqr: float
    skewness: float
    pearson_corr: float
    pearson_p_value: float
    spearman_corr: float
    outlier_count: int
    outlier_percentage: float

class DatasetSummaryResponse(BaseModel):
    raw_shape: ShapeInfo
    cleaned_shape: ShapeInfo
    duplicates_removed: int
    missing_values: Dict[str, int]
    outliers: OutliersInfo
    columns: List[str]
    target_variable: str
    numeric_stats: Dict[str, Dict[str, Any]]
    categorical_stats: Dict[str, Dict[str, Any]]
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    numerical_analysis: List[NumericalFeatureStat] = Field(default_factory=list)
    target_leakage_audit: Dict[str, Any] = Field(default_factory=dict)

class DatasetPreviewResponse(BaseModel):
    dataset_type: str
    page: int
    page_size: int
    total_records: int
    total_pages: int
    columns: List[str]
    records: List[Dict[str, Any]]
