from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SelectKBestItem(BaseModel):
    feature: str
    original_column: Optional[str] = None
    feature_type: Optional[str] = "Numerical"
    score: float
    p_value: float
    selected: bool
    rank: int
    normalized_pct: float

class CorrelationItem(BaseModel):
    feature: str
    original_column: Optional[str] = None
    feature_type: Optional[str] = "Numerical"
    correlation: float
    abs_correlation: float
    relationship: str
    strength: str = "Moderate"
    spearman_rho: float = 0.0
    p_value: float = 0.0
    selected: bool = False
    rank: Optional[int] = 0
    decision_reason: str = ""

class MutualInfoItem(BaseModel):
    feature: str
    original_column: Optional[str] = None
    feature_type: Optional[str] = "Numerical"
    mi_score: float
    normalized_pct: float
    rank: int
    selected: Optional[bool] = False

class RFImportanceItem(BaseModel):
    feature: str
    importance: float
    percentage: float

class FeatureChipItem(BaseModel):
    name: str
    selected: bool
    reason: str

class MulticollinearityPair(BaseModel):
    feature_1: str
    feature_2: str
    correlation: float
    abs_correlation: float
    status: str
    explanation: str

class VIFItem(BaseModel):
    feature: str
    vif: float
    r_squared: float
    interpretation: str
    warning: bool
    feature_type: Optional[str] = "Numerical"

class MethodComparisonItem(BaseModel):
    feature: str
    original_column: Optional[str] = None
    feature_type: Optional[str] = "Numerical"
    pearson_r: float
    pearson_abs_r: Optional[float] = None
    pearson_p_value: Optional[float] = None
    spearman_rho: Optional[float] = None
    correlation_rank: Optional[int] = None
    direction: Optional[str] = "Positive"
    selected_in_correlation: bool
    f_score: float
    f_pvalue: Optional[float] = None
    select_k_rank: Optional[int] = None
    selected_in_f_test: bool
    mi_score: float
    mi_rank: Optional[int] = None
    selected_in_mi: Optional[bool] = False
    consensus_rank: Optional[int] = None
    consensus_votes: Optional[int] = 0
    consensus_score: Optional[str] = None
    agreement: str
    is_selected: Optional[bool] = False

class ScatterPoint(BaseModel):
    x: float
    y: float

class ScatterTrendItem(BaseModel):
    feature: str
    points: List[Dict[str, float]]
    slope: float
    intercept: float
    r_squared: float

class CorrelationHeatmaps(BaseModel):
    columns: List[str]
    features: Optional[List[str]] = None
    pearson: List[List[float]]
    pearson_matrix: Optional[List[List[float]]] = None
    spearman: List[List[float]]
    spearman_matrix: Optional[List[List[float]]] = None

class FeatureSelectionResponse(BaseModel):
    k: int
    total_candidate_features: int
    numerical_features_count: Optional[int] = 7
    categorical_features_count: Optional[int] = 6
    encoded_categorical_features_count: Optional[int] = 26
    selected_features_count: int
    selected_features: List[str]
    select_k_best: List[SelectKBestItem]
    correlation_ranking: List[CorrelationItem]
    mutual_information: List[MutualInfoItem] = Field(default_factory=list)
    random_forest_importance: List[RFImportanceItem]
    feature_chips: List[FeatureChipItem]
    selection_method_note: str
    multicollinearity_analysis: List[MulticollinearityPair] = Field(default_factory=list)
    vif_analysis: List[VIFItem] = Field(default_factory=list)
    vif_selected: List[VIFItem] = Field(default_factory=list)
    comparison_table: List[MethodComparisonItem] = Field(default_factory=list)
    overlapping_features: List[str] = Field(default_factory=list)
    correlation_unique: List[str] = Field(default_factory=list)
    f_test_unique: List[str] = Field(default_factory=list)
    heatmaps: Optional[CorrelationHeatmaps] = None
    scatter_trends: List[ScatterTrendItem] = Field(default_factory=list)
