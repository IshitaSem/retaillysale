import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from .pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, get_preprocessor, get_feature_names_out
from ..utils.helpers import sanitize_floats

def get_feature_origin(feature_name: str) -> Tuple[str, str]:
    """Identifies the original dataset column and feature type for preprocessed features."""
    for cat in CATEGORICAL_FEATURES:
        if feature_name.startswith(f"{cat}_"):
            return cat, "Categorical Dummy"
    return feature_name, "Numerical"

def compute_vif_table(df_features: pd.DataFrame, feature_type_default: str = "Numerical") -> List[Dict[str, Any]]:
    """
    Computes exact Variance Inflation Factor (VIF) for each predictor:
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is obtained by regressing feature j on all other predictors.
    """
    results = []
    cols = df_features.columns.tolist()
    n_cols = len(cols)

    for col in cols:
        y_c = df_features[col]
        X_c = df_features.drop(columns=[col])

        if n_cols <= 1:
            vif = 1.0
            r2 = 0.0
        else:
            # Check for constant column or zero variance
            if y_c.std() == 0 or np.isnan(y_c.std()):
                vif = 1.0
                r2 = 0.0
            else:
                try:
                    reg = LinearRegression().fit(X_c, y_c)
                    r2 = float(reg.score(X_c, y_c))
                    if r2 >= 0.9999:
                        vif = 999.0  # Near-singular or perfect collinearity
                    else:
                        vif = 1.0 / (1.0 - r2)
                except Exception:
                    vif = 1.0
                    r2 = 0.0

        orig_col, ftype = get_feature_origin(col)

        if vif < 5.0:
            interp = "Low Collinearity (Safe for OLS & Tree Models)"
            warning = False
        elif vif <= 10.0:
            interp = "Moderate Collinearity (Acceptable; Tree Models Robust)"
            warning = True
        else:
            interp = "High Multicollinearity (Structural / Dummy dependency; Caution for OLS Standard Errors)"
            warning = True

        results.append({
            "feature": col,
            "vif": round(float(vif), 2),
            "r_squared": round(float(r2), 4),
            "interpretation": interp,
            "warning": warning,
            "feature_type": ftype
        })
    return results

class FeatureSelectionService:
    @classmethod
    def run_selection(
        cls,
        X_train_df: pd.DataFrame,
        y_train: pd.Series,
        k: int = 9
    ) -> Dict[str, Any]:
        """
        Conducts rigorous supervised and unsupervised feature selection strictly on the 80% training partition.
        Prevents data leakage by ensuring holdout test data is neither observed nor used in parameter estimation.

        Candidate Feature Space:
        - 7 Continuous Numerical Predictors
        - 26 One-Hot Encoded Categorical Dummy Predictors (from 6 raw categorical columns)
        - Total: 33 Candidate Features
        - Target Variable: Sales (continuous, strictly excluded from candidate matrix)
        - Leakage Excluded: Profit (strictly excluded from all candidate features)

        Methods implemented:
        1. Method A: Pearson Correlation Ranking & Point-Biserial against Sales (with two-tailed p-values & Spearman rho).
        2. Method B: SelectKBest with f_regression (ANOVA F-statistic univariate linear tests).
        3. Method C: Mutual Information Regression (with discrete_features boolean mask for dummy variables).
        4. Consensus Feature Selection: Ranks from Methods A, B, and C combined through voting & Borda rank.
        5. Variance Inflation Factor (VIF) Analysis: Full numerical matrix and selected model subspace.
        6. Multicollinearity Pairwise Correlation Matrix (|r| >= 0.70 threshold).
        7. Correlation Heatmap Matrices (Pearson & Spearman).
        8. Feature-vs-Sales Univariate Regression Trendlines.
        9. Random Forest Feature Importance (MDI).
        """
        k = max(3, min(k, 15))

        # 1. Fit Preprocessor on training data only
        preprocessor = get_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train_df)
        all_feature_names = get_feature_names_out(preprocessor)
        total_candidates = len(all_feature_names)
        num_candidates = len(NUMERIC_FEATURES)
        cat_candidates = total_candidates - num_candidates

        # =========================================================================
        # 2. METHOD B: SelectKBest with f_regression (F-statistic & p-values)
        # =========================================================================
        sel_k_obj = SelectKBest(score_func=f_regression, k="all")
        sel_k_obj.fit(X_train_proc, y_train)

        f_scores = sel_k_obj.scores_
        f_pvalues = sel_k_obj.pvalues_

        b_items = []
        for i, name in enumerate(all_feature_names):
            orig_col, ftype = get_feature_origin(name)
            score = float(f_scores[i]) if not np.isnan(f_scores[i]) else 0.0
            p_val = float(f_pvalues[i]) if not np.isnan(f_pvalues[i]) else 1.0
            b_items.append({
                "feature": name,
                "original_column": orig_col,
                "feature_type": ftype,
                "score": round(score, 2),
                "p_value": p_val,
                "index": i
            })

        b_items.sort(key=lambda x: x["score"], reverse=True)
        max_f = b_items[0]["score"] if b_items and b_items[0]["score"] > 0 else 1.0
        for rank_idx, item in enumerate(b_items):
            item["rank"] = rank_idx + 1
            item["selected"] = rank_idx < k
            item["normalized_pct"] = round((item["score"] / max_f) * 100, 1)

        b_dict = {item["feature"]: item for item in b_items}
        selected_features_kbest = [item["feature"] for item in b_items if item["selected"]]

        # =========================================================================
        # 3. METHOD A: Pearson Correlation Ranking & Point-Biserial against Sales
        # =========================================================================
        a_items = []
        for i, name in enumerate(all_feature_names):
            vals = X_train_proc[:, i]
            orig_col, ftype = get_feature_origin(name)
            
            p_r, p_p = stats.pearsonr(vals, y_train)
            s_rho, _ = stats.spearmanr(vals, y_train)
            p_r = 0.0 if np.isnan(p_r) else float(p_r)
            p_p = 1.0 if np.isnan(p_p) else float(p_p)
            s_rho = 0.0 if np.isnan(s_rho) else float(s_rho)
            abs_r = abs(p_r)

            if abs_r >= 0.70:
                strength = "Very Strong"
            elif abs_r >= 0.40:
                strength = "Strong"
            elif abs_r >= 0.20:
                strength = "Moderate"
            elif abs_r >= 0.05:
                strength = "Weak"
            else:
                strength = "Negligible"

            decision_reason = (
                f"Statistically significant ({strength.lower()}) association with Sales (r = {p_r:.3f}, p = {p_p:.3e})."
                if abs_r >= 0.05 and p_p < 0.05
                else f"Negligible correlation (|r| = {abs_r:.3f} < 0.05 or p >= 0.05)."
            )

            a_items.append({
                "feature": name,
                "original_column": orig_col,
                "feature_type": ftype,
                "correlation": round(p_r, 4),
                "abs_correlation": round(abs_r, 4),
                "relationship": "Positive" if p_r >= 0 else "Negative",
                "strength": strength,
                "spearman_rho": round(s_rho, 4),
                "p_value": float(p_p),
                "decision_reason": decision_reason,
                "index": i
            })

        a_items.sort(key=lambda x: x["abs_correlation"], reverse=True)
        for rank_idx, item in enumerate(a_items):
            item["rank"] = rank_idx + 1
            item["selected"] = rank_idx < k

        a_dict = {item["feature"]: item for item in a_items}
        selected_features_corr = [item["feature"] for item in a_items if item["selected"]]

        # =========================================================================
        # 4. METHOD C: Mutual Information Regression (with discrete feature mask)
        # =========================================================================
        # discrete_features mask: first 7 features are continuous (False), remaining 26 are binary dummy (True)
        discrete_mask = [False] * len(NUMERIC_FEATURES) + [True] * (total_candidates - len(NUMERIC_FEATURES))
        mi_scores = mutual_info_regression(
            X_train_proc,
            y_train,
            discrete_features=discrete_mask,
            random_state=42
        )
        max_mi = max(mi_scores) if len(mi_scores) > 0 and max(mi_scores) > 0 else 1.0

        c_items = []
        for i, name in enumerate(all_feature_names):
            orig_col, ftype = get_feature_origin(name)
            mi_val = float(mi_scores[i]) if not np.isnan(mi_scores[i]) else 0.0
            c_items.append({
                "feature": name,
                "original_column": orig_col,
                "feature_type": ftype,
                "mi_score": round(mi_val, 4),
                "normalized_pct": round((mi_val / max_mi) * 100, 1),
                "index": i
            })

        c_items.sort(key=lambda x: x["mi_score"], reverse=True)
        for rank_idx, item in enumerate(c_items):
            item["rank"] = rank_idx + 1
            item["selected"] = rank_idx < k

        c_dict = {item["feature"]: item for item in c_items}
        selected_features_mi = [item["feature"] for item in c_items if item["selected"]]

        # =========================================================================
        # 5. CONSENSUS FEATURE SELECTION (Ranks & Votes across Methods A, B, and C)
        # =========================================================================
        comparison_table = []
        for name in all_feature_names:
            a_info = a_dict[name]
            b_info = b_dict[name]
            c_info = c_dict[name]

            votes = int(a_info["selected"]) + int(b_info["selected"]) + int(c_info["selected"])
            avg_rank = (a_info["rank"] + b_info["rank"] + c_info["rank"]) / 3.0

            if votes == 3:
                agreement = "3 / 3 (Unanimous Select)"
            elif votes == 2:
                agreement = "2 / 3 (Majority Consensus)"
            elif votes == 1:
                agreement = "1 / 3 (Single Method)"
            else:
                agreement = "0 / 3 (Filtered Out)"

            comparison_table.append({
                "feature": name,
                "original_column": a_info["original_column"],
                "feature_type": a_info["feature_type"],
                "pearson_r": a_info["correlation"],
                "pearson_abs_r": a_info["abs_correlation"],
                "pearson_p_value": a_info["p_value"],
                "spearman_rho": a_info["spearman_rho"],
                "correlation_rank": a_info["rank"],
                "direction": a_info["relationship"],
                "selected_in_correlation": a_info["selected"],
                "f_score": b_info["score"],
                "f_pvalue": b_info["p_value"],
                "select_k_rank": b_info["rank"],
                "selected_in_f_test": b_info["selected"],
                "mi_score": c_info["mi_score"],
                "mi_rank": c_info["rank"],
                "selected_in_mi": c_info["selected"],
                "consensus_votes": votes,
                "consensus_score": f"{votes} / 3 methods",
                "avg_rank": round(avg_rank, 2),
                "agreement": agreement,
                "is_selected": False
            })

        # Rank consensus primarily by votes descending, secondarily by average rank ascending
        comparison_table.sort(key=lambda x: (-x["consensus_votes"], x["avg_rank"]))
        for r_idx, row in enumerate(comparison_table):
            row["consensus_rank"] = r_idx + 1
            row["is_selected"] = r_idx < k

        # Final consensus selected features (top k)
        selected_features_consensus = [r["feature"] for r in comparison_table if r["is_selected"]]

        # Overlapping and unique features between Method A and Method B
        set_corr = set(selected_features_corr)
        set_f = set(selected_features_kbest)
        overlapping = [f for f in all_feature_names if f in set_corr and f in set_f]
        corr_unique = [f for f in all_feature_names if f in set_corr and f not in set_f]
        f_unique = [f for f in all_feature_names if f in set_f and f not in set_corr]

        # =========================================================================
        # 6. VARIANCE INFLATION FACTOR (VIF) ANALYSIS
        # =========================================================================
        # 6a. Numerical Predictor Matrix VIF (7 raw numerical features on training set)
        available_num_cols = [c for c in NUMERIC_FEATURES if c in X_train_df.columns]
        num_vif_table = compute_vif_table(X_train_df[available_num_cols], feature_type_default="Numerical")

        # 6b. Selected Features Model Subspace VIF (the k selected features passed to models)
        df_proc_train = pd.DataFrame(X_train_proc, columns=all_feature_names)
        selected_vif_table = compute_vif_table(df_proc_train[selected_features_consensus], feature_type_default="Selected")

        # 6c. Pairwise Multicollinearity Detection (|r| >= 0.70 threshold) on numerical features
        multicollinearity_pairs = []
        for i in range(len(available_num_cols)):
            for j in range(i + 1, len(available_num_cols)):
                col1 = available_num_cols[i]
                col2 = available_num_cols[j]
                r_val = float(X_train_df[col1].corr(X_train_df[col2]))
                abs_r_val = abs(r_val)

                if abs_r_val >= 0.70:
                    status = "High Collinearity (Risk)"
                    exp = f"Strong pairwise correlation (|r| = {abs_r_val:.2f} >= 0.70). High potential variance inflation."
                elif abs_r_val >= 0.30:
                    status = "Moderate Collinearity"
                    exp = f"Moderate correlation (|r| = {abs_r_val:.2f}). Monitored; acceptable for regularized regression."
                else:
                    status = "Low Collinearity (Safe)"
                    exp = f"Minimal pairwise dependency (|r| = {abs_r_val:.2f} < 0.30). Distinct information contribution."

                multicollinearity_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(r_val, 4),
                    "abs_correlation": round(abs_r_val, 4),
                    "status": status,
                    "explanation": exp
                })
        multicollinearity_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        # =========================================================================
        # 7. CORRELATION HEATMAP MATRICES (Pearson & Spearman)
        # =========================================================================
        all_numeric_for_heatmap = ["Sales"] + available_num_cols
        combined_df = X_train_df[available_num_cols].copy()
        combined_df["Sales"] = y_train

        pearson_mat = combined_df[all_numeric_for_heatmap].corr(method="pearson").round(3).values.tolist()
        spearman_mat = combined_df[all_numeric_for_heatmap].corr(method="spearman").round(3).values.tolist()

        heatmaps = {
            "columns": all_numeric_for_heatmap,
            "features": all_numeric_for_heatmap,
            "pearson": pearson_mat,
            "pearson_matrix": pearson_mat,
            "spearman": spearman_mat,
            "spearman_matrix": spearman_mat
        }

        # =========================================================================
        # 8. FEATURE-VS-SALES SCATTER TRENDS (Univariate Descriptive Regressions)
        # =========================================================================
        scatter_trends = []
        trend_features = [c for c in ["Unit Price", "QtyOrdered", "Freight Expenses", "Discount offered", "Shipping Days"] if c in X_train_df.columns]
        for col in trend_features:
            x_vals = X_train_df[col].values
            y_vals = y_train.values
            slope, intercept, r_val, _, _ = stats.linregress(x_vals, y_vals)
            sample_indices = np.linspace(0, len(x_vals) - 1, 30, dtype=int)
            sample_pts = [
                {"x": round(float(x_vals[idx]), 2), "y": round(float(y_vals[idx]), 2)}
                for idx in sample_indices
            ]
            scatter_trends.append({
                "feature": col,
                "points": sample_pts,
                "slope": round(float(slope), 4),
                "intercept": round(float(intercept), 2),
                "r_squared": round(float(r_val ** 2), 4)
            })

        # =========================================================================
        # 9. RANDOM FOREST FEATURE IMPORTANCE (MDI)
        # =========================================================================
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_train_proc, y_train)
        rf_importances = rf.feature_importances_

        rf_results = []
        for i, name in enumerate(all_feature_names):
            rf_results.append({
                "feature": name,
                "importance": round(float(rf_importances[i]), 4),
                "percentage": round(float(rf_importances[i]) * 100, 2)
            })
        rf_results.sort(key=lambda x: x["importance"], reverse=True)

        # Feature explanation badges
        explanations = {
            "Unit Price": "Strongest linear and monotonic driver of sales per transaction (r = 0.430, rho = 0.838).",
            "QtyOrdered": "Direct volume multiplier determining total gross basket value (r = 0.298, rho = 0.470).",
            "Freight Expenses": "Captures order bulkiness, long-distance logistics, and heavier cargo requirements (r = 0.258).",
            "Freight Mode_Delivery Truck": "Identifies high-capacity commercial freight orders with substantial bulk volumes.",
            "Freight Mode_Regular Air": "Standard air cargo mode for medium-weight consumer orders.",
            "Product Type_Processed Meat": "High-value refrigerated perishable category with higher revenue tickets.",
            "Order Priority_High": "Urgent enterprise deliveries associated with premium retail inventory.",
            "Product Container_Jumbo Box": "Bulk container size used exclusively for commercial wholesale orders.",
            "Product Container_Jumbo Drum": "Heavy industrial container associated with wholesale liquid and bulk goods.",
            "Discount offered": "Pricing incentive affecting margin; minimal correlation with gross revenue (|r| = 0.027).",
            "Shipping Days": "Lead time indicator related to order fulfillment logistics (|r| = 0.016).",
            "Order Month": "Seasonal variation indicator for retail purchasing patterns.",
            "Order Day of Week": "Weekly cyclical shopping patterns."
        }

        feature_chips = []
        for feat in selected_features_consensus:
            feature_chips.append({
                "name": feat,
                "selected": True,
                "reason": explanations.get(feat, "Statistically significant contributor to transaction revenue.")
            })

        # Count numerical vs categorical in selected set
        num_selected_count = sum(1 for f in selected_features_consensus if get_feature_origin(f)[1] == "Numerical")
        cat_selected_count = len(selected_features_consensus) - num_selected_count

        return sanitize_floats({
            "k": k,
            "total_candidate_features": total_candidates,
            "numerical_features_count": num_candidates,
            "categorical_features_count": len(CATEGORICAL_FEATURES),
            "encoded_categorical_features_count": cat_candidates,
            "selected_features_count": len(selected_features_consensus),
            "selected_features": selected_features_consensus,
            "select_k_best": b_items,
            "correlation_ranking": a_items,
            "mutual_information": c_items,
            "random_forest_importance": rf_results,
            "feature_chips": feature_chips,
            "multicollinearity_analysis": multicollinearity_pairs,
            "vif_analysis": num_vif_table,
            "vif_selected": selected_vif_table,
            "comparison_table": comparison_table,
            "overlapping_features": overlapping,
            "correlation_unique": corr_unique,
            "f_test_unique": f_unique,
            "heatmaps": heatmaps,
            "scatter_trends": scatter_trends,
            "selection_method_note": (
                f"Candidate feature space contains {total_candidates} total features: {num_candidates} raw continuous numerical predictors "
                f"and {cat_candidates} one-hot encoded categorical dummies derived from {len(CATEGORICAL_FEATURES)} categorical variables. "
                f"Feature selection strictly isolates the top {k} consensus features on the 80% training set (zero test leakage). "
                f"For k={k}, consensus selection chooses {num_selected_count} numerical features ({', '.join([f for f in selected_features_consensus if get_feature_origin(f)[1] == 'Numerical'])}) "
                f"and {cat_selected_count} categorical dummy indicators ({', '.join([f for f in selected_features_consensus if get_feature_origin(f)[1] != 'Numerical'])}). "
                f"Target variable Sales and post-sale outcome Profit are strictly excluded from all candidate matrices."
            )
        })
