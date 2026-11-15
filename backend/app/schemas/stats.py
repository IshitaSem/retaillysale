from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MannWhitneyGroup(BaseModel):
    name: str
    count: int
    median: float
    median_formatted: str
    mean: float
    mean_formatted: str

class MannWhitneyTestInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    p_value_formatted: str
    alpha: float
    reject_null: bool
    group_a: MannWhitneyGroup
    group_b: MannWhitneyGroup
    interpretation: str
    effect_size: Optional[float] = None
    effect_size_label: Optional[str] = None

class SpearmanTestInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    p_value_formatted: str
    alpha: float
    reject_null: bool
    sample_size: int
    interpretation: str

class ContingencyTableInfo(BaseModel):
    categories_x: List[str]
    categories_y: List[str]
    matrix: List[List[int]]
    expected_matrix: Optional[List[List[float]]] = None

class ChiSquareTestInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    p_value_formatted: str
    alpha: float
    reject_null: bool
    contingency_table: ContingencyTableInfo
    cramers_v: Optional[float] = None
    cramers_v_interpretation: Optional[str] = None
    sparse_cells_warning: Optional[bool] = False
    interpretation: str

class TukeyHSDComparison(BaseModel):
    group_1: str
    group_2: str
    mean_diff: float
    p_value: float
    lower_ci: float
    upper_ci: float
    reject_null: bool

class GroupSummaryStat(BaseModel):
    name: str
    count: int
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float

class AnovaTestInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    degrees_of_freedom_between: Optional[int] = None
    degrees_of_freedom_within: Optional[int] = None
    p_value: float
    p_value_formatted: str
    alpha: float
    reject_null: bool
    eta_squared: Optional[float] = None
    levene_statistic: Optional[float] = None
    levene_p_value: Optional[float] = None
    homogeneity_satisfied: Optional[bool] = None
    tukey_hsd: Optional[List[TukeyHSDComparison]] = None
    group_stats: Optional[List[GroupSummaryStat]] = None
    interpretation: str

class TTestInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    group_a_name: str
    group_b_name: str
    group_a_mean: float
    group_b_mean: float
    group_a_n: Optional[int] = None
    group_a_std: Optional[float] = None
    group_b_n: Optional[int] = None
    group_b_std: Optional[float] = None
    null_hypothesis: str
    alternative_hypothesis: str
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    p_value_formatted: str
    alpha: float
    equal_variance_assumed: bool
    cohens_d: float
    effect_size_label: str
    effect_size_interpretation: Optional[str] = None
    reject_null: bool
    is_significant: Optional[bool] = None
    conclusion: Optional[str] = None
    interpretation: str

class KruskalWallisInfo(BaseModel):
    test_name: str
    type: str
    variables: List[str]
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    degrees_of_freedom: int
    p_value: float
    p_value_formatted: str
    alpha: float
    eta_squared: float
    reject_null: bool
    interpretation: str

class CorrelationSignificanceItem(BaseModel):
    feature: str
    target: str
    pearson_r: float
    pearson_t_stat: float
    pearson_p_value: float
    pearson_reject_null: bool
    spearman_rho: float
    spearman_p_value: float
    spearman_reject_null: bool
    sample_size: int

class MultipleTestingItem(BaseModel):
    test_id: str
    test_name: str
    comparison: str
    raw_p_value: float
    original_p_value: Optional[float] = None
    rank: int
    bh_critical_value: float
    adjusted_p_value: float
    significant_raw: bool
    significant_fdr: bool
    rejected_null: Optional[bool] = None

class MultipleTestingSummary(BaseModel):
    method: str
    fdr_alpha: float
    nominal_alpha: Optional[float] = None
    total_tests: int
    significant_raw_count: int
    significant_fdr_count: int
    significant_after_fdr: Optional[int] = None
    tests: List[MultipleTestingItem]

class StatsResponse(BaseModel):
    mann_whitney: MannWhitneyTestInfo
    spearman: SpearmanTestInfo
    chi_square: ChiSquareTestInfo
    anova: AnovaTestInfo
    all_chi_square_tests: List[ChiSquareTestInfo] = Field(default_factory=list)
    all_anova_tests: List[AnovaTestInfo] = Field(default_factory=list)
    t_tests: List[TTestInfo] = Field(default_factory=list)
    kruskal_wallis_tests: List[KruskalWallisInfo] = Field(default_factory=list)
    correlation_significance_tests: List[CorrelationSignificanceItem] = Field(default_factory=list)
    multiple_testing_correction: Optional[MultipleTestingSummary] = None
    assumption_summary: Dict[str, Any] = Field(default_factory=dict)
