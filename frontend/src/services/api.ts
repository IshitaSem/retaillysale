/**
 * RetaillySale — Centralized API Client Service
 * Connects frontend React components to the Python FastAPI backend.
 */

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  // If explicitly configured for production (e.g. Render), use that
  if (envUrl && !envUrl.includes('localhost') && !envUrl.includes('127.0.0.1')) {
    return envUrl.replace(/\/$/, '');
  }
  // On local machines, force 127.0.0.1 to avoid Windows IPv6 [::1] connection refused issues
  if (typeof window !== 'undefined' && window.location) {
    const host = window.location.hostname;
    if (!host || host === 'localhost' || host === '127.0.0.1') {
      return 'http://127.0.0.1:8081';
    }
    return `http://${host}:8081`;
  }
  return 'http://127.0.0.1:8081';
}

export const API_BASE_URL = getApiBaseUrl();

// ================= DATASET & CLEANING TYPES =================

export interface NumericalAnalysisItem {
  feature: string;
  mean: number;
  median: number;
  std: number;
  variance: number;
  min: number;
  max: number;
  q25: number;
  q75: number;
  iqr: number;
  skewness: number;
  pearson_corr?: number;
  pearson_corr_with_sales?: number;
  pearson_p_value: number;
  spearman_corr?: number;
  spearman_corr_with_sales?: number;
  outlier_count?: number;
  outliers_count?: number;
  outlier_percentage?: number;
  outliers_pct?: number;
}

export interface TargetLeakageAudit {
  audited_feature: string;
  is_leaked?: boolean;
  leakage_risk?: string;
  leakage_type?: string;
  mathematical_proof?: string;
  decision: string;
  reason?: string;
  academic_rationale?: string;
  correlation_with_target?: number;
}

export interface DatasetSummary {
  raw_shape: { rows: number; columns: number };
  cleaned_shape: { rows: number; columns: number };
  duplicates_removed: number;
  missing_values: Record<string, number>;
  outliers: {
    column: string;
    q1: number;
    q3: number;
    iqr: number;
    lower_bound: number;
    upper_bound: number;
    count: number;
    percentage: number;
  };
  columns: string[];
  numerical_features?: string[];
  categorical_features?: string[];
  target_variable: string;
  numeric_stats: Record<string, Record<string, number>>;
  categorical_stats: Record<string, { unique_count: number; top_values: Record<string, number> }>;
  numerical_analysis?: NumericalAnalysisItem[];
  target_leakage_audit?: TargetLeakageAudit;
}

export interface DatasetPreview {
  dataset_type: string;
  page: number;
  page_size: number;
  total_records: number;
  total_pages: number;
  columns: string[];
  records: Record<string, any>[];
}

// ================= EDA TYPES =================

export interface KPIInfo {
  total_sales: number;
  total_sales_formatted: string;
  avg_daily_sales: number;
  avg_daily_sales_formatted: string;
  avg_order_sales: number;
  avg_order_sales_formatted: string;
  segment_count: number;
  city_count: number;
  records_analyzed: number;
}

export interface TrendPoint {
  YearMonth: string;
  total_sales: number;
  avg_sales: number;
  order_count: number;
}

export interface StoreMixItem {
  name: string;
  total_sales: number;
  percentage: number;
  color: string;
}

export interface PromoBar {
  tier: string;
  avg_sales: number;
  order_count: number;
}

export interface DistributionBar {
  bin_start: number;
  bin_end: number;
  count: number;
  normalized_height: number;
}

export interface CorrelationRow {
  feature: string;
  correlations: number[];
}

export interface BoxplotStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  lower_whisker: number;
  upper_whisker: number;
}

export interface EDAResponse {
  kpis: KPIInfo;
  sales_trend: {
    monthly: TrendPoint[];
    chart_values: number[];
    dates: string[];
  };
  store_mix: StoreMixItem[];
  promo_comparison: PromoBar[];
  sales_distribution: DistributionBar[];
  correlation_matrix: {
    columns: string[];
    matrix: CorrelationRow[];
  };
  boxplots: Record<string, BoxplotStats>;
  product_mix: { product_type: string; total_sales: number; order_count: number }[];
  pipeline_status: Record<string, string>;
}

// ================= STATISTICAL TESTS TYPES =================

export interface MannWhitneyGroup {
  name: string;
  count: number;
  median: number;
  median_formatted: string;
  mean: number;
  mean_formatted: string;
}

export interface MannWhitneyTestInfo {
  test_name: string;
  type: string;
  variables: string[];
  null_hypothesis: string;
  alternative_hypothesis: string;
  statistic: number;
  p_value: number;
  p_value_formatted: string;
  alpha: number;
  reject_null: boolean;
  group_a: MannWhitneyGroup;
  group_b: MannWhitneyGroup;
  interpretation: string;
}

export interface SpearmanTestInfo {
  test_name: string;
  type: string;
  variables: string[];
  null_hypothesis: string;
  alternative_hypothesis: string;
  statistic: number;
  p_value: number;
  p_value_formatted: string;
  alpha: number;
  reject_null: boolean;
  sample_size: number;
  interpretation: string;
}

export interface ContingencyCell {
  observed: number;
  expected: number;
}

export interface ChiSquareTestInfo {
  test_name: string;
  type: string;
  variables: string[];
  null_hypothesis: string;
  alternative_hypothesis: string;
  statistic: number;
  degrees_of_freedom: number;
  p_value: number;
  p_value_formatted: string;
  alpha: number;
  reject_null: boolean;
  cramers_v?: number;
  effect_size_interpretation?: string;
  contingency_table: {
    categories_x: string[];
    categories_y: string[];
    matrix: number[][];
    expected_matrix?: number[][];
  };
  interpretation: string;
}

export interface TukeyComparisonItem {
  comparison?: string;
  group_1?: string;
  group_2?: string;
  group_a?: string;
  group_b?: string;
  mean_diff: number;
  p_value: number;
  reject_null: boolean;
  lower_ci?: number;
  upper_ci?: number;
  ci_lower?: number;
  ci_upper?: number;
}

export interface AnovaGroupStat {
  group?: string;
  name?: string;
  count: number;
  mean: number;
  std: number;
  median?: number;
  ci_lower?: number;
  ci_upper?: number;
}

export interface AnovaTestInfo {
  test_name: string;
  type: string;
  variables: string[];
  null_hypothesis: string;
  alternative_hypothesis: string;
  statistic: number;
  p_value: number;
  p_value_formatted: string;
  alpha: number;
  reject_null: boolean;
  df_between?: number;
  df_within?: number;
  degrees_of_freedom_between?: number;
  degrees_of_freedom_within?: number;
  eta_squared?: number;
  group_stats?: AnovaGroupStat[];
  levene_statistic?: number;
  levene_p_value?: number;
  homogeneity_satisfied?: boolean;
  tukey_hsd?: TukeyComparisonItem[];
  tukey_comparisons?: TukeyComparisonItem[];
  interpretation: string;
}

export interface TTestItem {
  test_name: string;
  variable: string;
  group_a_name: string;
  group_a_n: number;
  group_a_mean: number;
  group_a_std: number;
  group_b_name: string;
  group_b_n: number;
  group_b_mean: number;
  group_b_std: number;
  t_statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  cohens_d: number;
  effect_size_interpretation: string;
  is_significant: boolean;
  conclusion: string;
}

export interface KruskalWallisItem {
  test_name: string;
  dependent_variable: string;
  grouping_variable: string;
  h_statistic: number;
  p_value: number;
  is_significant: boolean;
  interpretation: string;
}

export interface CorrelationSigItem {
  feature: string;
  pearson_r: number;
  pearson_t_stat: number;
  pearson_p_value: number;
  pearson_significant: boolean;
  spearman_rho: number;
  spearman_t_stat: number;
  spearman_p_value: number;
  spearman_significant: boolean;
  interpretation: string;
}

export interface FDRTestRow {
  test_name: string;
  original_p_value: number;
  rank: number;
  bh_critical_value: number;
  rejected_null: boolean;
}

export interface MultipleTestingSummary {
  method: string;
  nominal_alpha: number;
  total_tests: number;
  significant_before_fdr: number;
  significant_after_fdr: number;
  tests: FDRTestRow[];
}

export interface AssumptionSummary {
  shapiro_wilk?: {
    test?: string;
    statistic: number;
    p_value: number;
    verdict?: string;
    note?: string;
  };
  levene_variance?: {
    test?: string;
    statistic: number;
    p_value: number;
    verdict?: string;
  };
  shapiro_normality?: {
    test_name?: string;
    statistic: number;
    p_value: number;
    is_normal?: boolean;
    conclusion?: string;
  };
  levene_homoscedasticity?: {
    test_name?: string;
    statistic: number;
    p_value: number;
    is_homoscedastic?: boolean;
    conclusion?: string;
  };
}

export interface StatsResponse {
  mann_whitney: MannWhitneyTestInfo;
  spearman: SpearmanTestInfo;
  chi_square: ChiSquareTestInfo;
  anova: AnovaTestInfo;
  all_chi_square_tests?: ChiSquareTestInfo[];
  all_anova_tests?: AnovaTestInfo[];
  t_tests?: TTestItem[];
  kruskal_wallis_tests?: KruskalWallisItem[];
  correlation_significance_tests?: CorrelationSigItem[];
  multiple_testing_correction?: MultipleTestingSummary;
  assumption_summary?: AssumptionSummary;
}

// ================= FEATURE SELECTION TYPES =================

export interface SelectKBestItem {
  feature: string;
  score: number;
  p_value: number;
  selected: boolean;
  rank: number;
  normalized_pct: number;
}

export interface CorrelationItem {
  feature: string;
  correlation: number;
  abs_correlation: number;
  relationship: string;
}

export interface RFImportanceItem {
  feature: string;
  importance: number;
  percentage: number;
}

export interface MutualInfoItem {
  feature: string;
  mi_score: number;
  normalized_pct: number;
  rank: number;
}

export interface MulticollinearityPair {
  feature_1?: string;
  feature_2?: string;
  feature_x?: string;
  feature_y?: string;
  correlation?: number;
  abs_correlation?: number;
  pearson_r?: number;
  abs_r?: number;
  severity?: string;
  status?: string;
  is_multicollinear?: boolean;
  recommendation?: string;
  explanation?: string;
}

export interface VIFItem {
  feature: string;
  vif: number;
  r_squared: number;
  interpretation: string;
  warning: boolean;
  feature_type?: string;
}

export interface MethodComparisonItem {
  feature: string;
  original_column?: string;
  feature_type?: string;
  pearson_r?: number;
  pearson_abs_r?: number;
  pearson_p_value?: number;
  spearman_rho?: number;
  f_score?: number;
  f_pvalue?: number;
  mi_score?: number;
  selected_in_correlation?: boolean;
  selected_in_f_test?: boolean;
  selected_in_mi?: boolean;
  agreement?: string;
  correlation_rank?: number;
  correlation_abs_r?: number;
  select_k_rank?: number;
  select_k_score?: number;
  mi_rank?: number;
  is_selected_by_f_test?: boolean;
  is_selected_by_corr?: boolean;
  is_selected_by_mi?: boolean;
  consensus_score?: string;
  consensus_rank?: number;
  consensus_votes?: number;
  is_selected?: boolean;
  direction?: string;
}

export interface CorrelationHeatmaps {
  features: string[];
  pearson_matrix: number[][];
  spearman_matrix: number[][];
}

export interface ScatterTrendPoint {
  x: number;
  y: number;
}

export interface ScatterTrendItem {
  feature: string;
  slope: number;
  intercept: number;
  r_squared: number;
  trend_type: string;
  points: ScatterTrendPoint[];
}

export interface FeatureChipItem {
  name: string;
  selected: boolean;
  reason: string;
}

export interface FeatureSelectionResponse {
  k: number;
  total_candidate_features: number;
  numerical_features_count?: number;
  categorical_features_count?: number;
  encoded_categorical_features_count?: number;
  selected_features_count: number;
  selected_features: string[];
  select_k_best: SelectKBestItem[];
  correlation_ranking: CorrelationItem[];
  random_forest_importance: RFImportanceItem[];
  mutual_information?: MutualInfoItem[];
  multicollinearity_analysis?: MulticollinearityPair[];
  vif_analysis?: VIFItem[];
  vif_selected?: VIFItem[];
  comparison_table?: MethodComparisonItem[];
  overlapping_features?: string[];
  correlation_unique?: string[];
  f_test_unique?: string[];
  heatmaps?: CorrelationHeatmaps;
  scatter_trends?: ScatterTrendItem[];
  feature_chips: FeatureChipItem[];
  selection_method_note: string;
}

// ================= MODELS TYPES =================

export interface MetricValues {
  mae: number;
  mae_formatted: string;
  mse?: number;
  mse_formatted?: string;
  rmse: number;
  rmse_formatted: string;
  r2: number;
  r2_pct: number;
}

export interface ExperimentCard {
  id: string;
  model_name: string;
  feature_set: string;
  tone: string;
  badge: string;
  metrics: MetricValues;
}

export interface MetricImpact {
  all: number;
  selected: number;
  diff: number;
  status: 'improved' | 'unchanged' | 'reduced';
}

export interface ModelImpact {
  MAE: MetricImpact;
  RMSE: MetricImpact;
  R2: MetricImpact;
}

export interface BestModelInfo {
  best_model: string;
  reason: string;
}

export interface ActualVsPredictedPoint {
  index?: number;
  actual: number;
  lr_pred?: number;
  rf_pred?: number;
  predicted_lr?: number;
  predicted_rf?: number;
  residual_lr?: number;
  residual_rf?: number;
  unit_price?: number;
  qty_ordered?: number;
}

export interface ResidualModelStats {
  mean_error: number;
  std_error: number;
  max_error: number;
  skewness: number;
  points?: { predicted?: number; residual?: number; index?: number; actual?: number; error?: number }[];
}

export interface ResidualDiagnostics {
  linear_regression?: ResidualModelStats;
  random_forest?: ResidualModelStats;
  mean_error_lr?: number;
  mean_error_rf?: number;
  std_error_lr?: number;
  std_error_rf?: number;
  max_error_lr?: number;
  max_error_rf?: number;
  skewness_lr?: number;
  skewness_rf?: number;
  points?: any[];
}

export interface ModelComparisonResponse {
  k: number;
  trained_at: string;
  train_samples: number;
  test_samples: number;
  experiments: ExperimentCard[];
  chart_heights: Record<string, number[]>;
  selected_features: string[];
  impact: {
    linear_regression: ModelImpact;
    random_forest: ModelImpact;
  };
  random_forest_importance: { feature: string; importance: number; percentage: number }[];
  best_model: BestModelInfo;
  actual_vs_predicted?: ActualVsPredictedPoint[];
  residuals?: ResidualDiagnostics;
}

export interface ModelStatusResponse {
  is_trained: boolean;
  models_available: string[];
  artifacts_path: string;
  last_trained?: string;
}

export interface PredictionRequest {
  unit_price: number;
  qty_ordered: number;
  discount_offered?: number;
  freight_expenses?: number;
  shipping_days?: number;
  order_month?: number;
  order_day_of_week?: string;
  order_priority?: string;
  freight_mode?: string;
  segment?: string;
  product_type?: string;
  product_container?: string;
  region?: string;
  model_type?: string;
  use_selected_features?: boolean;
}

export interface PredictionResponse {
  predicted_sales: number;
  predicted_sales_formatted: string;
  model_used: string;
  feature_subset: string;
  inputs: Record<string, any>;
  explanation: string;
}

// ================= API CLIENT =================

async function executeFetch<T>(targetUrl: string, options: RequestInit, headers: Record<string, string>): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);

  try {
    const res = await fetch(targetUrl, {
      ...options,
      signal: controller.signal,
      headers,
    });

    if (!res.ok) {
      let errorMessage = `HTTP Error ${res.status} (${res.statusText})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) {
          errorMessage = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // ignore parse error
      }
      throw new Error(errorMessage);
    }

    return (await res.json()) as T;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;

  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...((options.headers as Record<string, string>) || {}),
  };

  if (options.body && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  try {
    return await executeFetch<T>(url, options, headers);
  } catch (primaryErr: any) {
    // Resilient fallback 1: try http://127.0.0.1:8081 if not already attempted
    if (typeof window !== 'undefined' && !url.includes('127.0.0.1:8081')) {
      try {
        const fallbackUrl = `http://127.0.0.1:8081${endpoint}`;
        return await executeFetch<T>(fallbackUrl, options, headers);
      } catch {
        // continue to next fallback
      }
    }

    // Resilient fallback 2: try relative Vite proxy endpoint
    if (typeof window !== 'undefined' && url.startsWith('http')) {
      try {
        return await executeFetch<T>(endpoint, options, headers);
      } catch {
        // continue
      }
    }

    console.error(`API Request failed for ${endpoint}:`, primaryErr);
    throw primaryErr;
  }
}

export const api = {
  checkHealth: () => apiRequest<{ status: string; dataset_ready: boolean; models_ready: boolean }>('/health'),
  getDatasetSummary: () => apiRequest<DatasetSummary>('/api/dataset/summary'),
  getDatasetPreview: (type: 'raw' | 'cleaned' = 'cleaned', page = 1, pageSize = 10, search = '') => {
    const params = new URLSearchParams({
      dataset_type: type,
      page: String(page),
      page_size: String(pageSize),
    });
    if (search) params.append('search', search);
    return apiRequest<DatasetPreview>(`/api/dataset/preview?${params.toString()}`);
  },
  getEDA: () => apiRequest<EDAResponse>('/api/eda'),
  getStatisticalTests: () => apiRequest<StatsResponse>('/api/statistical-tests'),
  getFeatureSelection: (k = 6) => apiRequest<FeatureSelectionResponse>(`/api/feature-selection?k=${k}`),
  getModelComparison: (k = 6) => apiRequest<ModelComparisonResponse>(`/api/models/compare?k=${k}`),
  getModelStatus: () => apiRequest<ModelStatusResponse>('/api/models/status'),
  predictSales: (payload: PredictionRequest) =>
    apiRequest<PredictionResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};

export default api;
