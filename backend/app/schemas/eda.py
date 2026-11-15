from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class KPIInfo(BaseModel):
    total_sales: float
    total_sales_formatted: str
    avg_daily_sales: float
    avg_daily_sales_formatted: str
    avg_order_sales: float
    avg_order_sales_formatted: str
    segment_count: int
    city_count: int
    records_analyzed: int

class TrendPoint(BaseModel):
    YearMonth: str
    total_sales: float
    avg_sales: float
    order_count: int

class SalesTrendInfo(BaseModel):
    monthly: List[TrendPoint]
    chart_values: List[float]
    dates: List[str]

class StoreMixItem(BaseModel):
    name: str
    total_sales: float
    percentage: float
    color: str

class PromoBar(BaseModel):
    tier: str
    avg_sales: float
    order_count: int

class DistributionBar(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    normalized_height: float

class CorrelationRow(BaseModel):
    feature: str
    correlations: List[float]

class CorrelationMatrixInfo(BaseModel):
    columns: List[str]
    matrix: List[CorrelationRow]

class BoxplotStats(BaseModel):
    min: float
    q1: float
    median: float
    q3: float
    max: float
    lower_whisker: float
    upper_whisker: float

class ProductMixItem(BaseModel):
    product_type: str
    total_sales: float
    order_count: int

class EDAResponse(BaseModel):
    kpis: KPIInfo
    sales_trend: SalesTrendInfo
    store_mix: List[StoreMixItem]
    promo_comparison: List[PromoBar]
    sales_distribution: List[DistributionBar]
    correlation_matrix: CorrelationMatrixInfo
    boxplots: Dict[str, BoxplotStats]
    product_mix: List[ProductMixItem]
    pipeline_status: Dict[str, str]
