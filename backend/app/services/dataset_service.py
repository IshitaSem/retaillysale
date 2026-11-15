import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from scipy import stats

from ..config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from ..utils.helpers import sanitize_floats, format_inr

class DatasetService:
    _instance: Optional["DatasetService"] = None
    _raw_df: Optional[pd.DataFrame] = None
    _cleaned_df: Optional[pd.DataFrame] = None

    @classmethod
    def get_instance(cls) -> "DatasetService":
        if cls._instance is None:
            cls._instance = DatasetService()
        return cls._instance

    def __init__(self):
        self.load_data()

    def load_data(self) -> None:
        """Load raw dataset and build cleaned dataset."""
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

        # Load raw
        if str(RAW_DATA_PATH).endswith(".xlsx") or str(RAW_DATA_PATH).endswith(".xls"):
            self._raw_df = pd.read_excel(RAW_DATA_PATH)
        else:
            self._raw_df = pd.read_csv(RAW_DATA_PATH)

        # Build cleaned dataset
        self._cleaned_df = self._clean_data(self._raw_df)

        # Save to processed directory if not exists or out of date
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    def _clean_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Reproducible preprocessing pipeline without data leakage."""
        df = df_raw.copy()

        # 1. Deduplication
        df = df.drop_duplicates()

        # 2. DateTime features
        if "Order Date" in df.columns:
            df["Order Date"] = pd.to_datetime(df["Order Date"])
            df["Order Year"] = df["Order Date"].dt.year
            df["Order Month"] = df["Order Date"].dt.month
            df["Order Quarter"] = df["Order Date"].dt.quarter
            df["Order Day of Week"] = df["Order Date"].dt.day_of_week.astype(int)
            df["Order Day Name"] = df["Order Date"].dt.day_name()

        if "Ship Date" in df.columns:
            df["Ship Date"] = pd.to_datetime(df["Ship Date"])

        if "Order Date" in df.columns and "Ship Date" in df.columns:
            df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
            df["Shipping Days"] = df["Shipping Days"].clip(lower=0)

        # 3. Numeric validity checks
        for num_col in ["Sales", "Unit Price", "QtyOrdered"]:
            if num_col in df.columns:
                df = df[df[num_col] > 0]

        if "Discount offered" in df.columns:
            df["Discount offered"] = df["Discount offered"].clip(lower=0.0, upper=1.0)

        if "Freight Expenses" in df.columns:
            df["Freight Expenses"] = df["Freight Expenses"].clip(lower=0.0)

        return df

    def get_raw_data(self) -> pd.DataFrame:
        if self._raw_df is None:
            self.load_data()
        return self._raw_df

    def get_cleaned_data(self) -> pd.DataFrame:
        if self._cleaned_df is None:
            self.load_data()
        return self._cleaned_df

    def get_numerical_features(self) -> List[str]:
        """Automatically identify numerical predictor features, excluding target Sales and leakage variable Profit."""
        clean_df = self.get_cleaned_data()
        excluded = {"Sales", "Profit", "Order Year", "Order Quarter"}
        return [
            col for col in clean_df.select_dtypes(include=[np.number]).columns
            if col not in excluded
        ]

    def get_categorical_features(self) -> List[str]:
        """Automatically identify categorical predictor features."""
        clean_df = self.get_cleaned_data()
        excluded = {"Order Date", "Ship Date", "Order Day Name", "Country"}
        return [
            col for col in clean_df.select_dtypes(include=["object", "str"]).columns
            if col not in excluded
        ]

    def get_target_leakage_audit(self) -> Dict[str, Any]:
        """
        Audit post-transaction variables (e.g. Profit) for target leakage.
        Profit is mathematically defined as: Profit = Sales - COGS - Logistics - Discount.
        Because Profit cannot be known at order quotation time before transaction settlement,
        including Profit as a predictor causes severe data leakage.
        """
        clean_df = self.get_cleaned_data()
        profit_corr = float(clean_df["Profit"].corr(clean_df["Sales"])) if "Profit" in clean_df.columns else 0.0
        return {
            "audited_feature": "Profit",
            "correlation_with_target": round(profit_corr, 4),
            "leakage_risk": "Critical Target Leakage",
            "decision": "Excluded from All Model Inputs and Feature Selection",
            "reason": (
                "Profit is a post-sale financial settlement metric computed as (Sales - Expenses). "
                "In a predictive quoting scenario, Profit is an unobservable future outcome at order creation time. "
                "Training models with Profit creates artificial lookahead bias and guarantees catastrophic failure in live deployment."
            )
        }

    def get_summary(self) -> Dict[str, Any]:
        raw_df = self.get_raw_data()
        clean_df = self.get_cleaned_data()

        # Outlier count via IQR on Sales
        q1 = float(clean_df["Sales"].quantile(0.25))
        q3 = float(clean_df["Sales"].quantile(0.75))
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        outliers_count = int(((clean_df["Sales"] < lower_bound) | (clean_df["Sales"] > upper_bound)).sum())

        # Missing values
        missing_summary = {col: int(clean_df[col].isnull().sum()) for col in clean_df.columns}

        # Numeric statistics
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        num_stats = {}
        for col in numeric_cols:
            num_stats[col] = {
                "mean": round(float(clean_df[col].mean()), 4),
                "std": round(float(clean_df[col].std()), 4),
                "variance": round(float(clean_df[col].var()), 4),
                "min": round(float(clean_df[col].min()), 4),
                "q25": round(float(clean_df[col].quantile(0.25)), 4),
                "median": round(float(clean_df[col].median()), 4),
                "q75": round(float(clean_df[col].quantile(0.75)), 4),
                "max": round(float(clean_df[col].max()), 4),
                "skewness": round(float(clean_df[col].skew()), 4),
            }

        # Detailed Numerical Predictor Analysis Table
        numerical_predictors = self.get_numerical_features()
        numerical_analysis_table = []
        for col in numerical_predictors:
            s = clean_df[col]
            q25_c = float(s.quantile(0.25))
            q75_c = float(s.quantile(0.75))
            iqr_c = q75_c - q25_c
            lower_c = q25_c - 1.5 * iqr_c
            upper_c = q75_c + 1.5 * iqr_c
            col_outliers = int(((s < lower_c) | (s > upper_c)).sum())
            p_r, p_p = stats.pearsonr(s, clean_df["Sales"])
            s_rho, _ = stats.spearmanr(s, clean_df["Sales"])

            numerical_analysis_table.append({
                "feature": col,
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std": round(float(s.std()), 2),
                "variance": round(float(s.var()), 2),
                "min": round(float(s.min()), 2),
                "q25": round(q25_c, 2),
                "q75": round(q75_c, 2),
                "max": round(float(s.max()), 2),
                "iqr": round(iqr_c, 2),
                "skewness": round(float(s.skew()), 3),
                "pearson_corr": round(float(p_r), 4),
                "pearson_p_value": float(p_p),
                "spearman_corr": round(float(s_rho), 4),
                "outlier_count": col_outliers,
                "outlier_percentage": round((col_outliers / len(clean_df)) * 100, 2)
            })

        # Categorical statistics
        cat_cols = clean_df.select_dtypes(include=["object", "str"]).columns.tolist()
        cat_stats = {}
        for col in cat_cols:
            val_counts = clean_df[col].value_counts().head(5).to_dict()
            cat_stats[col] = {
                "unique_count": int(clean_df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in val_counts.items()}
            }

        return sanitize_floats({
            "raw_shape": {"rows": len(raw_df), "columns": len(raw_df.columns)},
            "cleaned_shape": {"rows": len(clean_df), "columns": len(clean_df.columns)},
            "duplicates_removed": int(raw_df.duplicated().sum()),
            "missing_values": missing_summary,
            "outliers": {
                "column": "Sales",
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": outliers_count,
                "percentage": round((outliers_count / len(clean_df)) * 100, 2)
            },
            "columns": list(clean_df.columns),
            "target_variable": "Sales",
            "numerical_features": numerical_predictors,
            "categorical_features": self.get_categorical_features(),
            "numerical_analysis": numerical_analysis_table,
            "target_leakage_audit": self.get_target_leakage_audit(),
            "numeric_stats": num_stats,
            "categorical_stats": cat_stats
        })

    def get_preview(
        self,
        dataset_type: str = "cleaned",
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        df = self.get_cleaned_data() if dataset_type.lower() == "cleaned" else self.get_raw_data()
        
        # Format datetimes as string for JSON serialization
        df_preview = df.copy()
        for col in df_preview.select_dtypes(include=["datetime64[ns]", "datetime64[us]", "datetime"]):
            df_preview[col] = df_preview[col].dt.strftime("%Y-%m-%d")

        if search:
            search_str = search.lower()
            mask = df_preview.astype(str).apply(lambda row: row.str.lower().str.contains(search_str).any(), axis=1)
            df_preview = df_preview[mask]

        total_records = len(df_preview)
        total_pages = max(1, (total_records + page_size - 1) // page_size)
        current_page = max(1, min(page, total_pages))
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = df_preview.iloc[start_idx:end_idx]

        records = paged_df.to_dict(orient="records")

        return sanitize_floats({
            "dataset_type": dataset_type,
            "page": current_page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": total_pages,
            "columns": list(df.columns),
            "records": records
        })
