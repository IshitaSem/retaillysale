"""RetaillySale — Standalone Dataset and Modeling Diagnostic Script
Run this script to inspect raw and cleaned datasets, statistical tests, feature selection, and model performance.
"""
import sys
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.services.dataset_service import DatasetService
from app.services.eda_service import EDAService
from app.services.stats_service import StatsService
from app.ml.trainer import ModelTrainer
from app.ml.predictor import SalesPredictor

def main():
    print("=" * 60)
    print("  RETAILLYSALE: M.Sc. DATA SCIENCE WORKFLOW DIAGNOSTIC")
    print("=" * 60)

    # 1. Dataset Summary
    print("\n[1] DATASET INSPECTION & CLEANING:")
    ds = DatasetService.get_instance()
    summary = ds.get_summary()
    print(f"  Raw records:     {summary['raw_shape']['rows']:,} rows, {summary['raw_shape']['columns']} columns")
    print(f"  Cleaned records: {summary['cleaned_shape']['rows']:,} rows, {summary['cleaned_shape']['columns']} columns")
    print(f"  Duplicates:      {summary['duplicates_removed']}")
    print(f"  Sales Outliers:  {summary['outliers']['count']} records ({summary['outliers']['percentage']}%)")

    # 2. EDA Metrics
    print("\n[2] EXPLORATORY DATA ANALYSIS (EDA):")
    eda = EDAService.get_eda_data()
    kpis = eda["kpis"]
    print(f"  Total Sales:     {kpis['total_sales_formatted']}")
    print(f"  Avg Daily Sales: {kpis['avg_daily_sales_formatted']}")
    print(f"  Avg Order Sales: {kpis['avg_order_sales_formatted']}")
    print("  Segment Breakdown:")
    for s in eda["store_mix"]:
        print(f"    - {s['name']:25s}: {s['percentage']}% (₹{s['total_sales']:,.2f})")

    # 3. Statistical Testing
    print("\n[3] STATISTICAL HYPOTHESIS TESTING:")
    tests = StatsService.get_statistical_tests()
    mw = tests["mann_whitney"]
    print(f"  * {mw['test_name']}:")
    print(f"    Statistic = {mw['statistic']}, p-value = {mw['p_value_formatted']} (Reject H0: {mw['reject_null']})")
    print(f"    Interpretation: {mw['interpretation']}")

    sp = tests["spearman"]
    print(f"\n  * {sp['test_name']}:")
    print(f"    rho = {sp['statistic']}, p-value = {sp['p_value_formatted']} (Reject H0: {sp['reject_null']})")
    print(f"    Interpretation: {sp['interpretation']}")

    chi = tests["chi_square"]
    print(f"\n  * {chi['test_name']}:")
    print(f"    chi2 = {chi['statistic']}, p-value = {chi['p_value_formatted']}, dof = {chi['degrees_of_freedom']} (Reject H0: {chi['reject_null']})")
    print(f"    Interpretation: {chi['interpretation']}")

    # 4. Model Comparison
    print("\n[4] MODEL TRAINING & FEATURE SELECTION EXPERIMENTS:")
    comp = ModelTrainer.train_and_save(k=6)
    print(f"  Top 6 Selected Features: {', '.join(comp['selected_features'])}")
    print("\n  Experiment Results (Holdout Test Set):")
    for exp in comp["experiments"]:
        m = exp["metrics"]
        print(f"    [{exp['id']}] {exp['model_name']:20s} ({exp['feature_set']:12s}) -> MAE: {m['mae_formatted']}, RMSE: {m['rmse_formatted']}, R²: {m['r2']:.4f}")

    print(f"\n  Best Model: {comp['best_model']['best_model']}")
    print(f"  Analysis:   {comp['best_model']['reason']}")

    # 5. Prediction Verification
    print("\n[5] LIVE PREDICTION VERIFICATION:")
    test_pred = SalesPredictor.predict(
        unit_price=45.50,
        qty_ordered=8,
        discount_offered=0.05,
        freight_expenses=7.50,
        freight_mode="Express Air",
        segment="Restaurant Chain",
        product_type="Processed Meat",
        order_priority="High"
    )
    print(f"  Input: Unit Price=₹45.50, Qty=8, Discount=5%, Freight=₹7.50 ({test_pred['inputs']['freight_mode']})")
    print(f"  Prediction: {test_pred['predicted_sales_formatted']} using {test_pred['model_used']}")
    print(f"  Summary:    {test_pred['explanation']}")
    print("\n" + "=" * 60)
    print("  ALL DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
