import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "RetaillySale"
    assert data["status"] == "online"
    assert "endpoints" in data

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["dataset_ready"] is True
    assert data["models_ready"] is True

def test_dataset_summary():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["raw_shape"]["rows"] == 2534
    assert data["cleaned_shape"]["rows"] == 2534
    assert data["target_variable"] == "Sales"
    assert "outliers" in data
    assert "numeric_stats" in data
    assert "categorical_stats" in data

def test_dataset_preview():
    # Test cleaned preview
    response = client.get("/api/dataset/preview?page=1&page_size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert len(data["records"]) == 5
    assert data["total_records"] == 2534

    # Test raw preview
    response_raw = client.get("/api/dataset/preview?dataset_type=raw&page=1&page_size=5")
    assert response_raw.status_code == 200
    data_raw = response_raw.json()
    assert len(data_raw["records"]) == 5

def test_eda():
    response = client.get("/api/eda")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert data["kpis"]["total_sales"] > 0
    assert "sales_trend" in data
    assert len(data["sales_trend"]["chart_values"]) > 0
    assert "store_mix" in data
    assert len(data["store_mix"]) == 4
    assert "promo_comparison" in data
    assert "correlation_matrix" in data

def test_statistical_tests():
    response = client.get("/api/statistical-tests")
    assert response.status_code == 200
    data = response.json()
    assert "mann_whitney" in data
    assert "spearman" in data
    assert "chi_square" in data
    assert "anova" in data
    assert data["spearman"]["reject_null"] is True
    assert data["mann_whitney"]["statistic"] > 0

def test_feature_selection():
    response = client.get("/api/feature-selection?k=6")
    assert response.status_code == 200
    data = response.json()
    assert data["k"] == 6
    assert len(data["selected_features"]) == 6
    assert len(data["select_k_best"]) > 0
    assert len(data["correlation_ranking"]) > 0
    assert len(data["random_forest_importance"]) > 0

def test_models_compare():
    response = client.get("/api/models/compare?k=6")
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) == 4
    assert "impact" in data
    assert "chart_heights" in data
    assert "best_model" in data

def test_models_status():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is True
    assert len(data["models_available"]) == 4

def test_prediction_valid():
    payload = {
        "unit_price": 45.0,
        "qty_ordered": 5,
        "discount_offered": 0.05,
        "freight_expenses": 6.5,
        "shipping_days": 2,
        "order_month": 6,
        "order_day_of_week": "Wednesday",
        "order_priority": "High",
        "freight_mode": "Regular Air",
        "segment": "Restaurant Chain",
        "product_type": "Processed Meat",
        "product_container": "Small Box",
        "region": "West",
        "model_type": "Random Forest",
        "use_selected_features": True
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_sales"] > 0
    assert "₹" in data["predicted_sales_formatted"]
    assert "Random Forest" in data["model_used"]

def test_prediction_invalid():
    # Negative unit price should fail Pydantic validation with 422
    payload = {
        "unit_price": -10.0,
        "qty_ordered": 5
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422

def test_enhanced_dataset_summary_and_leakage_audit():
    response = client.get("/api/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert "numerical_features" in data
    assert "Sales" not in data["numerical_features"]
    assert "Profit" not in data["numerical_features"]
    assert "target_leakage_audit" in data
    assert data["target_leakage_audit"]["audited_feature"] == "Profit"
    assert "Excluded" in data["target_leakage_audit"]["decision"]
    assert len(data["numerical_analysis"]) > 0
    stat0 = data["numerical_analysis"][0]
    assert "mean" in stat0
    assert "median" in stat0
    assert "std" in stat0
    assert "variance" in stat0
    assert "skewness" in stat0
    assert "pearson_corr" in stat0

def test_enhanced_statistical_battery():
    response = client.get("/api/statistical-tests")
    assert response.status_code == 200
    data = response.json()
    # Chi-Square battery
    assert len(data["all_chi_square_tests"]) == 4
    for chi in data["all_chi_square_tests"]:
        assert "cramers_v" in chi
        assert chi["cramers_v"] >= 0.0
    # ANOVA battery with Tukey HSD
    assert len(data["all_anova_tests"]) >= 3
    product_anova = data["all_anova_tests"][0]
    assert product_anova["reject_null"] is True
    assert len(product_anova["tukey_hsd"]) > 0
    # Welch's T-Tests
    assert len(data["t_tests"]) == 2
    truck_ttest = data["t_tests"][1]
    assert truck_ttest["reject_null"] is True
    assert truck_ttest["cohens_d"] > 0
    # Kruskal-Wallis non-parametric
    assert len(data["kruskal_wallis_tests"]) == 3
    # Multiple testing correction
    fdr = data["multiple_testing_correction"]
    assert fdr["method"] == "Benjamini-Hochberg False Discovery Rate (FDR)"
    assert fdr["total_tests"] >= 10
    assert len(fdr["tests"]) == fdr["total_tests"]

def test_enhanced_feature_selection():
    response = client.get("/api/feature-selection?k=6")
    assert response.status_code == 200
    data = response.json()
    # Multicollinearity
    assert len(data["multicollinearity_analysis"]) > 0
    # Comparison table (Correlation vs F-test vs Mutual Info)
    assert len(data["comparison_table"]) > 0
    assert "agreement" in data["comparison_table"][0]
    # Mutual info
    assert len(data["mutual_information"]) > 0
    # Heatmaps
    assert data["heatmaps"] is not None
    assert len(data["heatmaps"]["columns"]) > 0
    assert len(data["heatmaps"]["pearson"]) == len(data["heatmaps"]["columns"])
    # Scatter trends
    assert len(data["scatter_trends"]) > 0
    assert "slope" in data["scatter_trends"][0]
    assert "intercept" in data["scatter_trends"][0]

def test_enhanced_models_comparison():
    response = client.get("/api/models/compare?k=6")
    assert response.status_code == 200
    data = response.json()
    # Check MSE in metrics
    for exp in data["experiments"]:
        assert "mse" in exp["metrics"]
        assert exp["metrics"]["mse"] > 0
    # Actual vs predicted coordinates
    assert len(data["actual_vs_predicted"]) > 0
    # Residuals
    assert "linear_regression" in data["residuals"]
    assert "random_forest" in data["residuals"]
    assert data["residuals"]["random_forest"]["std_error"] < data["residuals"]["linear_regression"]["std_error"]
