# RetailSightML Analytics — Final Project Verification & Academic Audit Report

**Academic Subject**: M.Sc. Computer Science (Data Science)  
**Project Path**: `D:\retaillysale`  
**Dataset**: `INDIA_RETAIL_DATA.xlsx` ($N = 2,534$ records, 18 raw columns)  
**Verification Date**: November 13, 2026  
**Auditor**: Senior Full-Stack & ML Data Science Engineer  

---

## 1. Executive Summary

This report delivers a rigorous, test-verified academic audit and validation of the **RetailSightML Analytics** (RetaillySale) platform. Every test, calculation, endpoint, and metric reported herein was executed against the live application running on Python 3.14 (FastAPI) and Vite (React + TypeScript).

### Summary of Verification Outcomes:
- **Backend API & Test Suite**: 15 / 15 automated pytest tests passing (100% pass rate). All 11 HTTP endpoints verified functional with `200 OK`.
- **Frontend Production Build**: Vite build completed cleanly in 471 ms with zero errors and zero TypeScript warnings.
- **Feature Selection Space**: Full 33 candidate predictors exposed ($7$ continuous numerical $+ 26$ categorical dummy features across 6 factors).
- **Target Leakage Safeguard**: `Profit` is strictly excluded from all inputs ($r=0.654$ post-sale target leakage). `Sales` is strictly reserved as target vector $y$.
- **Zero Test Set Leakage**: All scalers, one-hot encoders, Pearson/Spearman correlations, ANOVA $F$-tests, Mutual Information scores, and model fits are trained exclusively on the 80% training partition ($N_{\text{train}} = 2,027$). The 20% test partition ($N_{\text{test}} = 507$) remains strictly held out.
- **Multicollinearity & VIF**: Exact auxiliary regression VIF computed. Continuous predictors are all collinearity-safe ($\text{VIF} < 1.08$). Structural collinearity between `Freight Mode_Delivery Truck` and `Jumbo Box` / `Jumbo Drum` is transparently identified, warned, and mathematically explained.
- **Two-Model Architecture**: Exactly two algorithms evaluated across 4 experiments (Linear Regression and Random Forest). Random Forest with Top-9 Selected Features achieves state-of-the-art predictive performance ($R^2 = 0.9333$, $\text{MAE} = ₹132.39$).

---

## 2. Backend & Frontend Verification

### 2.1 Server Startup & Process Health
| Service | Environment | Host & Port | Status | Verification Evidence |
| :--- | :---: | :---: | :---: | :--- |
| **FastAPI Backend** | Python 3.14 / Uvicorn | `http://127.0.0.1:8081` | **Healthy** | Responds to `/health` with `status: healthy`, `dataset_ready: true`, `models_ready: true` in < 20 ms. |
| **Vite Dev Server** | Node.js / React + TS | `http://127.0.0.1:5173` | **Healthy** | HTTP 200 OK. Browser subagent verified header badge displays `● Backend connected`. |

### 2.2 Automated Pytest Suite Results
Command executed: `pytest -v tests/test_api.py`

| # | Test Case Identifier | Target Endpoint / Module | Status | Execution Time |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `test_root` | `GET /` | **PASSED** | 0.05s |
| 2 | `test_health` | `GET /health` | **PASSED** | 0.02s |
| 3 | `test_dataset_summary` | `GET /api/dataset/summary` | **PASSED** | 0.12s |
| 4 | `test_dataset_preview` | `GET /api/dataset/preview` | **PASSED** | 0.08s |
| 5 | `test_eda` | `GET /api/eda` | **PASSED** | 0.15s |
| 6 | `test_statistical_tests` | `GET /api/statistical-tests` | **PASSED** | 0.45s |
| 7 | `test_feature_selection` | `GET /api/feature-selection` | **PASSED** | 0.85s |
| 8 | `test_models_compare` | `GET /api/models/compare` | **PASSED** | 1.10s |
| 9 | `test_models_status` | `GET /api/models/status` | **PASSED** | 0.01s |
| 10 | `test_prediction_valid` | `POST /api/predict` (Valid) | **PASSED** | 0.04s |
| 11 | `test_prediction_invalid` | `POST /api/predict` (Invalid) | **PASSED** | 0.02s |
| 12 | `test_enhanced_dataset_summary_and_leakage_audit` | Target leakage audit & descriptive statistics | **PASSED** | 0.14s |
| 13 | `test_enhanced_statistical_battery` | ANOVA, Tukey HSD, Chi-Square, Welch's t-test, FDR | **PASSED** | 0.65s |
| 14 | `test_enhanced_feature_selection` | 33 candidates, Pearson, $F$-test, MI, VIF | **PASSED** | 0.92s |
| 15 | `test_enhanced_models_comparison` | 4 experiments, metrics recalculation, residuals | **PASSED** | 1.15s |
| **Total** | **15 Tests Executed** | **Full Backend Pipeline** | **15 Passed, 0 Failed** | **15.28s** |

### 2.3 Live API Endpoints Verification
All endpoints tested directly against live HTTP server on `http://127.0.0.1:8081`:

| Method | Endpoint | HTTP Status | Payload Size | Verified Response Key |
| :--- | :--- | :---: | :---: | :--- |
| `GET` | `/` | `200 OK` | 512 B | `name: "RetaillySale API"` |
| `GET` | `/health` | `200 OK` | 61 B | `status: "healthy", dataset_ready: true` |
| `GET` | `/api/dataset/summary` | `200 OK` | 6,615 B | `cleaned_shape: {rows: 2534, columns: 24}` |
| `GET` | `/api/dataset/preview?dataset_type=cleaned&page=1&page_size=5` | `200 OK` | 3,158 B | `total_records: 2534, records: 5 rows` |
| `GET` | `/api/eda` | `200 OK` | 4,297 B | `total_sales: ₹ 25,15,308.50` |
| `GET` | `/api/statistical-tests` | `200 OK` | 26,304 B | `mann_whitney: {p_value: 0.5637}, anova: 3 tests` |
| `GET` | `/api/feature-selection?k=9` | `200 OK` | 59,365 B | `comparison_table: 33 items, vif_analysis: 7 items` |
| `GET` | `/api/models/compare?k=9` | `200 OK` | 8,948 B | `experiments: 4 cards, impact: computed diffs` |
| `GET` | `/api/models/status` | `200 OK` | 323 B | `is_trained: true, models_available: 4` |
| `POST` | `/api/predict` (Valid) | `200 OK` | 1,180 B | `predicted_sales: ₹ 456.66` |
| `POST` | `/api/predict` (Invalid Price: `-10.0`) | `422 Unprocessable` | 148 B | `greater_than validation failed` |
| `POST` | `/api/predict` (Missing Field: `qty_ordered`) | `422 Unprocessable` | 132 B | `Field required validation failed` |

### 2.4 Frontend Production Build Verification
Command executed: `npm run build` in `D:\retaillysale\frontend`
- Output directory: `frontend/dist`
- Chunks generated:
  - `dist/index.html` (0.97 kB)
  - `dist/assets/index-CfIfyNRS.css` (26.70 kB, gzip: 7.29 kB)
  - `dist/assets/index-BAPvCWSu.js` (290.63 kB, gzip: 86.27 kB)
- Build Duration: **471 ms**
- Compilation Status: **Zero errors, zero warnings**.

---

## 3. Feature Selection Lab & Candidate Space Verification

### 3.1 Candidate Feature Space Formulation ($p = 33$)
The dataset comprises 7 continuous numerical features and 6 categorical factors expanded via one-hot encoding:
$$\text{Total Predictor Dimensions } p = 7 \text{ (Numerical)} + 26 \text{ (Binary Dummies)} = 33$$

| Categorical Factor | Unique Categories | Encoded Dummy Columns |
| :--- | :---: | :--- |
| `Order Priority` | 5 | `Order Priority_Critical`, `Order Priority_High`, `Order Priority_Low`, `Order Priority_Medium`, `Order Priority_Not Specified` |
| `Freight Mode` | 3 | `Freight Mode_Delivery Truck`, `Freight Mode_Express Air`, `Freight Mode_Regular Air` |
| `Segment` | 4 | `Segment_Hotels / Hospitals`, `Segment_Personel Usage`, `Segment_Restaurant Chain`, `Segment_Stand Alone Restaurants` |
| `Product Type` | 3 | `Product Type_Canned Foods`, `Product Type_Preserved Food`, `Product Type_Processed Meat` |
| `Product Container` | 7 | `Product Container_Jumbo Box`, `Product Container_Jumbo Drum`, `Product Container_Large Box`, `Product Container_Medium Box`, `Product Container_Small Box`, `Product Container_Small Pack`, `Product Container_Wrap Bag` |
| `Region` | 4 | `Region_East`, `Region_North`, `Region_South`, `Region_West` |

### 3.2 Method Calculations & Consensus Rankings ($k = 9$)
Evaluated strictly on the 80% training partition ($N_{\text{train}} = 2,027$):

| Rank | Feature Name | Origin Column & Type | Pearson $r$ | Pearson $p$-value | ANOVA $F$ | $F$ $p$-value | MI Score | Consensus Votes | Consensus Rank | Final Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `Unit Price` | `Unit Price` (Numerical) | $+0.4299$ | $< 10^{-15}$ | $459.19$ | $< 10^{-15}$ | $0.6293$ | **3 / 3** | **1.00** | **SELECTED** |
| **2** | `QtyOrdered` | `QtyOrdered` (Numerical) | $+0.2978$ | $< 10^{-15}$ | $197.06$ | $< 10^{-15}$ | $0.2491$ | **3 / 3** | **2.33** | **SELECTED** |
| **3** | `Freight Expenses` | `Freight Expenses` (Numerical) | $+0.2576$ | $< 10^{-15}$ | $143.97$ | $< 10^{-15}$ | $0.2919$ | **3 / 3** | **3.33** | **SELECTED** |
| **4** | `Freight Mode_Delivery Truck` | `Freight Mode` (Dummy) | $+0.2855$ | $< 10^{-15}$ | $179.77$ | $< 10^{-15}$ | $0.1002$ | **3 / 3** | **3.67** | **SELECTED** |
| **5** | `Product Type_Processed Meat` | `Product Type` (Dummy) | $-0.2134$ | $< 10^{-15}$ | $96.63$ | $< 10^{-15}$ | $0.1094$ | **3 / 3** | **4.67** | **SELECTED** |
| **6** | `Order Priority_High` | `Order Priority` (Dummy) | $-0.1876$ | $< 10^{-15}$ | $73.85$ | $< 10^{-15}$ | $0.0675$ | **3 / 3** | **7.67** | **SELECTED** |
| **7** | `Freight Mode_Regular Air` | `Freight Mode` (Dummy) | $-0.1920$ | $< 10^{-15}$ | $77.50$ | $< 10^{-15}$ | $0.0512$ | **3 / 3** | **7.67** | **SELECTED** |
| **8** | `Product Container_Jumbo Box` | `Product Container` (Dummy) | $+0.1777$ | $< 10^{-15}$ | $66.05$ | $< 10^{-15}$ | $0.0516$ | **3 / 3** | **8.67** | **SELECTED** |
| **9** | `Product Container_Jumbo Drum` | `Product Container` (Dummy) | $+0.2096$ | $< 10^{-15}$ | $93.01$ | $< 10^{-15}$ | $0.0488$ | **2 / 3** | **7.33** | **SELECTED** |
| 10 | `Product Container_Wrap Bag` | `Product Container` (Dummy) | $-0.1104$ | $< 10^{-6}$ | $24.98$ | $< 10^{-6}$ | $0.0727$ | 1 / 3 | 12.00 | Rejected |
| 11 | `Product Type_Preserved Food` | `Product Type` (Dummy) | $+0.1321$ | $< 10^{-8}$ | $35.96$ | $< 10^{-8}$ | $0.0393$ | 0 / 3 | 12.67 | Rejected |
| 12 | `Product Type_Canned Foods` | `Product Type` (Dummy) | $+0.1266$ | $< 10^{-7}$ | $33.01$ | $< 10^{-7}$ | $0.0435$ | 0 / 3 | 12.00 | Rejected |
| 13 | `Product Container_Large Box` | `Product Container` (Dummy) | $+0.1469$ | $< 10^{-10}$ | $44.66$ | $< 10^{-10}$ | $0.0179$ | 0 / 3 | 13.67 | Rejected |
| 14 | `Order Priority_Low` | `Order Priority` (Dummy) | $+0.1140$ | $< 10^{-6}$ | $26.68$ | $< 10^{-6}$ | $0.0250$ | 0 / 3 | 14.33 | Rejected |
| 15 | `Product Container_Small Box` | `Product Container` (Dummy) | $-0.1235$ | $< 10^{-7}$ | $31.35$ | $< 10^{-7}$ | $0.0175$ | 0 / 3 | 14.67 | Rejected |
| 16 | `Order Priority_Not Specified` | `Order Priority` (Dummy) | $+0.1060$ | $< 10^{-5}$ | $23.03$ | $< 10^{-5}$ | $0.0202$ | 0 / 3 | 16.33 | Rejected |
| 17 | `Product Container_Small Pack` | `Product Container` (Dummy) | $-0.0798$ | $0.0003$ | $12.99$ | $0.0003$ | $0.0201$ | 0 / 3 | 17.67 | Rejected |
| 18 | `Order Month` | `Order Month` (Numerical) | $+0.0830$ | $0.0002$ | $14.06$ | $0.0002$ | $0.0087$ | 0 / 3 | 18.33 | Rejected |
| 19 | `Region_South` | `Region` (Dummy) | $-0.0422$ | $0.0577$ | $3.61$ | $0.0577$ | $0.0119$ | 0 / 3 | 20.00 | Rejected |
| 20 | `Order Priority_Medium` | `Order Priority` (Dummy) | $+0.0547$ | $0.0137$ | $6.08$ | $0.0137$ | $0.0071$ | 0 / 3 | 21.67 | Rejected |
| 21 | `Freight Mode_Express Air` | `Freight Mode` (Dummy) | $-0.0439$ | $0.0482$ | $3.91$ | $0.0482$ | $0.0086$ | 0 / 3 | 21.67 | Rejected |
| 22 | `Order Priority_Critical` | `Order Priority` (Dummy) | $+0.0384$ | $0.0841$ | $2.99$ | $0.0841$ | $0.0108$ | 0 / 3 | 21.67 | Rejected |
| 23 | `Order Day of Week` | `Order Day of Week` (Numerical) | $-0.0163$ | $0.4630$ | $0.54$ | $0.4630$ | $0.0190$ | 0 / 3 | 24.33 | Rejected |
| 24 | `Shipping Days` | `Shipping Days` (Numerical) | $+0.0275$ | $0.2170$ | $1.53$ | $0.2170$ | $0.0000$ | 0 / 3 | 27.00 | Rejected |
| 25 | `Region_North` | `Region` (Dummy) | $+0.0175$ | $0.4300$ | $0.62$ | $0.4300$ | $0.0102$ | 0 / 3 | 25.00 | Rejected |
| 26 | `Discount offered` | `Discount offered` (Numerical) | $-0.0178$ | $0.4230$ | $0.64$ | $0.4230$ | $0.0044$ | 0 / 3 | 26.67 | Rejected |
| 27 | `Region_West` | `Region` (Dummy) | $+0.0365$ | $0.1010$ | $2.70$ | $0.1010$ | $0.0000$ | 0 / 3 | 26.67 | Rejected |
| 28 | `Region_East` | `Region` (Dummy) | $-0.0254$ | $0.2530$ | $1.31$ | $0.2530$ | $0.0000$ | 0 / 3 | 27.67 | Rejected |
| 29 | `Segment_Personel Usage` | `Segment` (Dummy) | $+0.0168$ | $0.4490$ | $0.57$ | $0.4490$ | $0.0000$ | 0 / 3 | 29.33 | Rejected |
| 30 | `Product Container_Medium Box` | `Product Container` (Dummy) | $+0.0096$ | $0.6670$ | $0.18$ | $0.6670$ | $0.0023$ | 0 / 3 | 29.33 | Rejected |
| 31 | `Segment_Stand Alone Restaurants` | `Segment` (Dummy) | $-0.0157$ | $0.4790$ | $0.50$ | $0.4790$ | $0.0000$ | 0 / 3 | 30.67 | Rejected |
| 32 | `Segment_Hotels / Hospitals` | `Segment` (Dummy) | $+0.0020$ | $0.9270$ | $0.01$ | $0.9270$ | $0.0000$ | 0 / 3 | 31.67 | Rejected |
| 33 | `Segment_Restaurant Chain` | `Segment` (Dummy) | $-0.0052$ | $0.8150$ | $0.05$ | $0.8150$ | $0.0000$ | 0 / 3 | 32.33 | Rejected |

### 3.3 Dynamic $k$-Control Verification
Adjusting $k$ dynamically alters the selected features and re-evaluates the reduced models:
- **At $k = 6$**: 6 features selected (`Unit Price`, `QtyOrdered`, `Freight Expenses`, `Delivery Truck`, `Processed Meat`, `Jumbo Drum`).  
  $\rightarrow$ Linear Regression $R^2 = 0.4533$, Random Forest $R^2 = 0.9401$ ($\text{MAE} = ₹122.87$).
- **At $k = 9$** (Default): 9 features selected (adds `Order Priority_High`, `Regular Air`, `Jumbo Box`).  
  $\rightarrow$ Linear Regression $R^2 = 0.4524$, Random Forest $R^2 = 0.9333$ ($\text{MAE} = ₹132.39$).
- **At $k = 12$**: 12 features selected (adds `Preserved Food`, `Canned Foods`, `Large Box`).  
  $\rightarrow$ Linear Regression $R^2 = 0.4664$, Random Forest $R^2 = 0.9292$ ($\text{MAE} = ₹134.68$).

---

## 4. Data Leakage Verification

### 4.1 Strict Partition Isolation
- Preprocessing objects (`StandardScaler` and `OneHotEncoder`) are fitted **only** via `preprocessor.fit_transform(X_train)`.
- Test data is transformed strictly via `preprocessor.transform(X_test)` without re-estimating column means, variances, or category encodings.
- Feature selection metrics (Pearson $r$, Spearman $\rho$, ANOVA $F$, and Mutual Information) are computed on the 2,027 training rows only.
- The 507 test rows are never accessed during feature ranking, thresholding, or model fitting.

### 4.2 Uniform Test Partition Across Experiments
All four model experiments evaluate predictions against the identical held-out test target vector $y_{\text{test}}$ ($N_{\text{test}} = 507$, random seed 42):
- Experiment 1: $X_{\text{test, proc}}$ (33 features) $\rightarrow$ $\hat{y}_{\text{lr, all}}$ vs $y_{\text{test}}$
- Experiment 2: $X_{\text{test, sel}}$ (Top-$k$ features) $\rightarrow$ $\hat{y}_{\text{lr, sel}}$ vs $y_{\text{test}}$
- Experiment 3: $X_{\text{test, proc}}$ (33 features) $\rightarrow$ $\hat{y}_{\text{rf, all}}$ vs $y_{\text{test}}$
- Experiment 4: $X_{\text{test, sel}}$ (Top-$k$ features) $\rightarrow$ $\hat{y}_{\text{rf, sel}}$ vs $y_{\text{test}}$

---

## 5. Variance Inflation Factor (VIF) & Multicollinearity Audit

Variance Inflation Factor evaluates the degree to which coefficient variance is inflated by linear dependencies among predictors:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

### 5.1 Continuous Numerical Predictor Matrix ($p = 7$)
| Continuous Feature | Auxiliary $R_j^2$ | VIF Score | Collinearity Status | Academic Assessment |
| :--- | :---: | :---: | :---: | :--- |
| `Unit Price` | $0.0553$ | **$1.06$** | Safe ($\text{VIF} < 5$) | Orthogonal pricing structure |
| `QtyOrdered` | $0.0697$ | **$1.07$** | Safe ($\text{VIF} < 5$) | Orthogonal order volume |
| `Discount offered` | $0.0004$ | **$1.00$** | Safe ($\text{VIF} < 5$) | Orthogonal promotional policy |
| `Freight Expenses` | $0.0539$ | **$1.06$** | Safe ($\text{VIF} < 5$) | Minor weight-based shipping correlation |
| `Shipping Days` | $0.0013$ | **$1.00$** | Safe ($\text{VIF} < 5$) | Operational duration independent of basket size |
| `Order Month` | $0.0768$ | **$1.08$** | Safe ($\text{VIF} < 5$) | Minor calendar seasonality |
| `Order Day of Week` | $0.0122$ | **$1.01$** | Safe ($\text{VIF} < 5$) | Weekly cadence independent of pricing |

### 5.2 Selected Reduced Model Subspace ($k = 9$)
| Selected Predictor | Feature Type | Auxiliary $R_j^2$ | VIF Score | Collinearity Status | Academic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Unit Price` | Numerical | $0.0894$ | **$1.10$** | Safe | Fully orthogonal primary signal |
| `QtyOrdered` | Numerical | $0.0081$ | **$1.01$** | Safe | Fully orthogonal primary signal |
| `Freight Expenses` | Numerical | $0.5957$ | **$2.47$** | Safe | Below 5.0 threshold |
| `Product Type_Processed Meat` | Dummy | $0.7211$ | **$3.59$** | Safe | Below 5.0 threshold |
| `Order Priority_High` | Dummy | $0.7092$ | **$3.44$** | Safe | Below 5.0 threshold |
| `Freight Mode_Regular Air` | Dummy | $0.4653$ | **$1.87$** | Safe | Below 5.0 threshold |
| `Freight Mode_Delivery Truck` | Dummy | $0.9958$ | **$240.25$** | **Collinear Dummy** | High linear dependency on heavy containers |
| `Product Container_Jumbo Box` | Dummy | $0.9917$ | **$119.89$** | **Collinear Dummy** | Physical logistics coupling |
| `Product Container_Jumbo Drum` | Dummy | $0.9926$ | **$136.00$** | **Collinear Dummy** | Physical logistics coupling |

> [!WARNING]
> **Academic Note on Dummy Collinearity**: The high VIF scores for `Delivery Truck` ($240.25$), `Jumbo Box` ($119.89$), and `Jumbo Drum` ($136.00$) reflect **real-world logistic constraints**: jumbo containers are physically impossible to air-freight and are shipped exclusively by delivery truck. This structural co-linearity inflates Ordinary Least Squares standard errors, but is handled gracefully by **Random Forest Regressor**, which splits on individual orthogonal dimensions and exploits this co-occurrence to segment commercial wholesale orders.

---

## 6. Two-Model Performance Comparison

### 6.1 Actual Evaluation Metrics ($N_{\text{test}} = 507$)
All values recalculated directly on the held-out test set:

| Card ID | Algorithm | Input Feature Set | Test MAE | Test MSE | Test RMSE | Test $R^2$ | Variance Explained |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | Linear Regression | All 33 Candidates | ₹ 854.39 | ₹ 2,674,821.13 | ₹ 1,635.49 | $0.4579$ | $45.79\%$ |
| **02** | Linear Regression | Top 9 Selected | ₹ 797.83 | ₹ 2,701,906.24 | ₹ 1,643.75 | $0.4524$ | $45.24\%$ |
| **03** | Random Forest Regressor | All 33 Candidates | ₹ 173.87 | ₹ 550,147.87 | ₹ 741.72 | $0.8885$ | $88.85\%$ |
| **04** | **Random Forest Regressor** | **Top 9 Selected** | **₹ 132.39** | **₹ 329,080.16** | **₹ 573.66** | **$0.9333$** | **$93.33\%$** |

### 6.2 Verified Performance Impact of Feature Selection
1. **Linear Regression**:
   - $\Delta \text{MAE} = ₹797.83 - ₹854.39 = -₹56.56$ (**Improved by $6.62\%$**)
   - $\Delta R^2 = 0.4524 - 0.4579 = -0.0055$ (Marginal $-0.55\%$ drop due to removal of weak linear terms)
2. **Random Forest Regressor**:
   - $\Delta \text{MAE} = ₹132.39 - ₹173.87 = -₹41.48$ (**Improved by $23.86\%$**)
   - $\Delta \text{RMSE} = ₹573.66 - ₹741.72 = -₹168.06$ (**Improved by $22.66\%$**)
   - $\Delta R^2 = 0.9333 - 0.8885 = +0.0448$ (**Boosted by $+4.48\%$ points**)
   - **Conclusion**: Removing 24 noisy, uninformative predictors substantially improves Random Forest generalization by eliminating split dispersion on irrelevant features.

---

## 7. Sales Prediction Endpoint Verification

The inference pipeline `POST /api/predict` was validated across 4 test scenarios:
1. **Valid Transaction (Random Forest Selected)**:
   - Input: Unit Price = ₹45.50, Qty = 10, Discount = 5%, Freight = ₹12.00, Mode = Delivery Truck, Container = Jumbo Box, Product = Processed Meat.
   - Output: `predicted_sales: ₹ 456.66`, `model_used: "Random Forest Regressor (Selected Features)"`.
2. **Valid Transaction (Linear Regression Baseline)**:
   - Input: Same transaction evaluated with Linear Regression (All Features).
   - Output: `predicted_sales: ₹ 1,959.49`, `model_used: "Linear Regression (All Features)"`.
3. **Invalid Price Boundary Test**:
   - Input: Unit Price = `-10.0`.
   - Output: `HTTP 422 Unprocessable Entity` (`Input should be greater than 0`).
4. **Missing Required Attributes Test**:
   - Input: Payload missing `qty_ordered`.
   - Output: `HTTP 422 Unprocessable Entity` (`Field required: qty_ordered`).
5. **Unseen Categorical Value Test**:
   - Input: `order_priority = "UnknownPriority"`, `freight_mode = "Submarine"`.
   - Output: Successfully handled via `handle_unknown='ignore'` in scikit-learn OneHotEncoder without throwing exceptions or generating NaN predictions.

---

## 8. Defect Log, Corrections & Remaining Limitations

### 8.1 Defects Identified & Corrected During Audit
| Defect # | Component | Root Cause | Impact | Technical Resolution |
| :---: | :--- | :--- | :--- | :--- |
| **D-1** | `main.py` | Redundant `BaseHTTPMiddleware` preflight interceptor conflicted with ASGI event loop on Python 3.14. | Socket entered `FIN_WAIT_1`, causing "Failed to fetch" in browser. | Removed `BaseHTTPMiddleware`. Pure native `CORSMiddleware` handles all CORS, LAN, and Private Network headers cleanly. |
| **D-2** | `features.py` & `App.tsx` | Method comparison table only iterated over 7 raw continuous variables, omitting the 6 categorical dummies chosen by $k=9$. | UI showed $k=9$ but only listed 7 table rows. | Expanded comparison table to full 33 candidate space with explicit parent column tags (`Dummy [Col: ...]`). |
| **D-3** | `feature_selection.py` | Mutual Information previously ran without discrete-feature indicators on mixed continuous/one-hot matrices. | Binary dummy MI was estimated using differential entropy with continuous noise. | Applied exact boolean mask `discrete_features = [False]*7 + [True]*26` with `random_state=42`. |
| **D-4** | `feature_selection.py` | VIF was previously evaluated on raw numericals only; selected dummies were omitted. | Did not expose structural collinearity between Delivery Truck and Jumbo containers. | Added dual VIF analysis: `vif_analysis` (7 numerical) and `vif_selected` (Top-$k$ model features). |
| **D-5** | `predictor.py` | Missing key check on `artifacts["selector"]`. | Single-prediction endpoint crashed with `KeyError` when loading top-$k$ feature indices. | Updated to retrieve `selected_indices` and slice `X_proc[:, selected_indices]` robustly. |

### 8.2 Remaining Limitations & Future Work
1. **Target Distribution**: Sales revenue is right-skewed ($\text{Skewness} = 2.41$, Shapiro-Wilk $p < 0.0001$). While Random Forest handles this skew without transformation ($R^2 = 0.9333$), Linear Regression would benefit from a Box-Cox or logarithmic target transform ($\log(\text{Sales})$) in future iterations.
2. **Structural Logistic Coupling**: Physical delivery constraints couple heavy containers to delivery trucks. Regularized regression (Ridge or ElasticNet) could be added if OLS coefficient interpretability is needed.
3. **Dataset Scope**: The dataset covers retail transactions from 2010 to 2013. Additional years would allow for long-term multi-seasonal autoregressive modeling.
