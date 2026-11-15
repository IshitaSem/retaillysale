from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "Unit Price",
    "QtyOrdered",
    "Discount offered",
    "Freight Expenses",
    "Shipping Days",
    "Order Month",
    "Order Day of Week"
]

CATEGORICAL_FEATURES = [
    "Order Priority",
    "Freight Mode",
    "Segment",
    "Product Type",
    "Product Container",
    "Region"
]

TARGET_FEATURE = "Sales"
LEAKAGE_EXCLUDED = ["Profit"]  # Post-sale outcome variable, strictly excluded to prevent target leakage

def get_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and one-hot encodes categoricals."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor

def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Extract flattened human-readable feature names after preprocessor is fitted."""
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_feature_names
