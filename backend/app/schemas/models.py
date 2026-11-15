from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MetricValues(BaseModel):
    mae: float
    mae_formatted: str
    mse: float = 0.0
    mse_formatted: str = ""
    rmse: float
    rmse_formatted: str
    r2: float
    r2_pct: float

class ExperimentCard(BaseModel):
    id: str
    model_name: str
    feature_set: str
    tone: str
    badge: str
    metrics: MetricValues

class MetricImpact(BaseModel):
    all: float
    selected: float
    diff: float
    status: str

class ModelImpact(BaseModel):
    MAE: MetricImpact
    MSE: Optional[MetricImpact] = None
    RMSE: MetricImpact
    R2: MetricImpact

class RFImportance(BaseModel):
    feature: str
    importance: float
    percentage: float

class BestModelInfo(BaseModel):
    best_model: str
    reason: str

class ActualVsPredictedPoint(BaseModel):
    actual: float
    lr_pred: float
    rf_pred: float
    unit_price: float
    qty_ordered: int

class ResidualStats(BaseModel):
    mean_error: float
    std_error: float
    max_error: float
    skewness: float
    points: List[Dict[str, float]] = Field(default_factory=list)

class ModelComparisonResponse(BaseModel):
    k: int
    trained_at: str
    train_samples: int
    test_samples: int
    experiments: List[ExperimentCard]
    chart_heights: Dict[str, List[float]]
    selected_features: List[str]
    impact: Dict[str, Any]
    random_forest_importance: List[RFImportance]
    best_model: BestModelInfo
    actual_vs_predicted: List[ActualVsPredictedPoint] = Field(default_factory=list)
    residuals: Dict[str, ResidualStats] = Field(default_factory=dict)

class ModelStatusResponse(BaseModel):
    is_trained: bool
    models_available: List[str]
    artifacts_path: str
    last_trained: Optional[str] = None
