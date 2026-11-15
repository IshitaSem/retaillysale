import os
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

from ..config import MODELS_DIR
from .trainer import ModelTrainer, ARTIFACTS_FILE
from .pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from ..utils.helpers import format_inr, sanitize_floats

class SalesPredictor:
    _artifacts: Optional[Dict[str, Any]] = None

    @classmethod
    def load_artifacts(cls) -> Dict[str, Any]:
        if cls._artifacts is None:
            if not ARTIFACTS_FILE.exists():
                ModelTrainer.train_and_save(k=6)
            cls._artifacts = joblib.load(ARTIFACTS_FILE)
        return cls._artifacts

    @classmethod
    def predict(
        cls,
        unit_price: float,
        qty_ordered: int,
        discount_offered: float = 0.05,
        freight_expenses: float = 6.05,
        shipping_days: int = 2,
        order_month: int = 6,
        order_day_of_week: str = "Wednesday",
        order_priority: str = "High",
        freight_mode: str = "Regular Air",
        segment: str = "Restaurant Chain",
        product_type: str = "Processed Meat",
        product_container: str = "Small Box",
        region: str = "West",
        model_type: str = "Random Forest",
        use_selected_features: bool = True
    ) -> Dict[str, Any]:
        """Generate sales prediction using trained scikit-learn pipeline."""
        artifacts = cls.load_artifacts()
        preprocessor = artifacts["preprocessor"]
        selector = artifacts.get("selector")
        models = artifacts["models"]

        # Validate inputs
        if unit_price <= 0:
            raise ValueError("Unit price must be strictly greater than 0.")
        if qty_ordered <= 0:
            raise ValueError("Quantity ordered must be at least 1.")
        if not (0.0 <= discount_offered <= 1.0):
            raise ValueError("Discount offered must be between 0.0 and 1.0 (e.g. 0.05 for 5%).")
        if freight_expenses < 0:
            raise ValueError("Freight expenses cannot be negative.")

        # Map day of week name to day integer if needed or string
        day_mapping = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        day_val = day_mapping.get(order_day_of_week, 2) if isinstance(order_day_of_week, str) else int(order_day_of_week)

        # Assemble single row DataFrame
        input_data = {
            "Unit Price": [float(unit_price)],
            "QtyOrdered": [int(qty_ordered)],
            "Discount offered": [float(discount_offered)],
            "Freight Expenses": [float(freight_expenses)],
            "Shipping Days": [int(shipping_days)],
            "Order Month": [int(order_month)],
            "Order Day of Week": [day_val],
            "Order Priority": [str(order_priority)],
            "Freight Mode": [str(freight_mode)],
            "Segment": [str(segment)],
            "Product Type": [str(product_type)],
            "Product Container": [str(product_container)],
            "Region": [str(region)],
        }
        df_input = pd.DataFrame(input_data)

        # Transform using fitted preprocessor
        X_proc = preprocessor.transform(df_input)

        # Select model
        is_rf = "forest" in model_type.lower()
        if use_selected_features:
            selected_indices = artifacts.get("selected_indices")
            if selected_indices is not None:
                X_sel = X_proc[:, selected_indices]
            elif "selector" in artifacts and artifacts["selector"] is not None:
                X_sel = artifacts["selector"].transform(X_proc)
            else:
                X_sel = X_proc
            model = models["rf_sel"] if is_rf else models["lr_sel"]
            chosen_model_name = "Random Forest Regressor (Selected Features)" if is_rf else "Linear Regression (Selected Features)"
            feature_subset_name = "Top-k Selected Features"
            prediction_val = float(model.predict(X_sel)[0])
        else:
            model = models["rf_all"] if is_rf else models["lr_all"]
            chosen_model_name = "Random Forest Regressor (All Features)" if is_rf else "Linear Regression (All Features)"
            feature_subset_name = "All Eligible Features"
            prediction_val = float(model.predict(X_proc)[0])

        # Enforce realistic sales floor
        prediction_val = max(0.0, prediction_val)

        return sanitize_floats({
            "predicted_sales": round(prediction_val, 2),
            "predicted_sales_formatted": format_inr(prediction_val),
            "model_used": chosen_model_name,
            "feature_subset": feature_subset_name,
            "inputs": {
                "unit_price": unit_price,
                "qty_ordered": qty_ordered,
                "discount_offered": discount_offered,
                "freight_expenses": freight_expenses,
                "shipping_days": shipping_days,
                "order_month": order_month,
                "order_day_of_week": order_day_of_week,
                "order_priority": order_priority,
                "freight_mode": freight_mode,
                "segment": segment,
                "product_type": product_type,
                "product_container": product_container,
                "region": region,
                "model_type": model_type
            },
            "explanation": (
                f"Prediction of {format_inr(prediction_val)} produced by {chosen_model_name}. "
                f"The primary revenue drivers are Unit Price (₹{unit_price:,.2f}) and Quantity Ordered ({qty_ordered} units), "
                f"adjusted for shipping logistics ({freight_mode}, ₹{freight_expenses:,.2f}) and promotional discount ({discount_offered*100:.1f}%)."
            )
        })
