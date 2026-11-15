# RetaillySale — Retail Sales Prediction & Feature Selection Platform

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF.svg)](https://vitejs.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E.svg)](https://scikit-learn.org)

**RetaillySale** is an interactive, full-stack Machine Learning application developed for the **M.Sc. Computer Science (Data Science)** curriculum. It demonstrates a complete, leak-free machine learning workflow for retail sales prediction, feature selection, and statistical hypothesis testing on real-world Indian retail transaction data.

---

## 1. Project Overview & Workflow

The platform covers an end-to-end academic data science pipeline:

1. **Dataset Loading & Exploration**: Ingests `INDIA_RETAIL_DATA.xlsx` (2,534 records, 18 initial columns).
2. **Data Cleaning & Preprocessing**: Deduplication, positive value constraints, outlier flagging (IQR method: 322 orders flagged), and date feature engineering (`Shipping Days`, `Order Month`, `Order Day of Week`, `Order Quarter`). Excludes post-sale `Profit` to prevent target leakage.
3. **Exploratory Data Analysis (EDA)**: Dynamic KPIs (₹25,15,308.50 total sales), monthly sales trends, customer segment breakdown (Hotels/Hospitals 33.4%, Personal 26.5%, Restaurant Chain 23.6%, Stand Alone 16.5%), discount tier comparisons, and numerical correlation matrix.
4. **Statistical Hypothesis Testing**:
   - **Mann–Whitney U Test**: Non-parametric test comparing high-discount ($\ge 5\%$) vs low-discount ($< 5\%$) orders ($U = 786,109.50, p = 0.5637$ — Fail to reject $H_0$).
   - **Spearman Rank Correlation**: Monotonic association between Unit Price and Sales ($\rho = 0.8400, p < 0.0001$ — Reject $H_0$).
   - **Chi-Square Test of Independence**: Association between Order Priority and Freight Logistics Mode ($\chi^2 = 332.16, df = 8, p < 0.0001$ — Reject $H_0$).
   - **One-Way ANOVA**: Sales variance across customer segments ($F = 0.5102, p = 0.6753$ — Fail to reject $H_0$).
5. **Feature Selection**:
   - **Filter Method**: `SelectKBest` with `f_regression` (ANOVA F-statistic).
   - **Correlation Method**: Pearson & Spearman correlation with target `Sales`.
   - **Model-Based Method**: Random Forest Mean Decrease in Impurity (MDI).
   - Dynamically configurable $k \in [3, 10]$ (default $k=6$).
6. **Machine Learning Model Comparison**:
   - Compares **Linear Regression** (baseline) vs **Random Forest Regressor** (nonlinear).
   - 4-way validation experiments on an identical 20% holdout test set (507 observations, seed = 42).
   - **Results**: Feature selection dramatically boosts Random Forest performance:
     - Linear Regression (All Features): MAE = ₹854.39, RMSE = ₹1,635.49, $R^2 = 0.4579$
     - Linear Regression (Selected Features): MAE = ₹789.02, RMSE = ₹1,642.32, $R^2 = 0.4533$
     - Random Forest (All Features): MAE = ₹173.87, RMSE = ₹741.72, $R^2 = 0.8885$
     - **Random Forest (Selected Features)**: **MAE = ₹122.87, RMSE = ₹543.61, $R^2 = 0.9401$**
7. **Interactive Prediction & Visualizations**: Real-time sales forecasting with trained pipelines, interactive parameter controls, and formatted INR currency outputs.

---

## 2. Repository Structure

```
retaillysale/
├── backend/
│   ├── app/
│   │   ├── api/                     # REST API routers
│   │   │   ├── routes_dataset.py    # /api/dataset/summary, /api/dataset/preview
│   │   │   ├── routes_eda.py        # /api/eda
│   │   │   ├── routes_features.py   # /api/feature-selection
│   │   │   ├── routes_models.py     # /api/models/compare, /api/models/status
│   │   │   ├── routes_predict.py    # /api/predict
│   │   │   └── routes_stats.py      # /api/statistical-tests
│   │   ├── ml/                      # Machine learning pipelines
│   │   │   ├── feature_selection.py # SelectKBest & correlation ranking
│   │   │   ├── pipeline.py          # ColumnTransformer & feature definitions
│   │   │   ├── predictor.py         # Inference handler
│   │   │   └── trainer.py           # Experiment trainer & evaluator
│   │   ├── schemas/                 # Pydantic request/response models
│   │   ├── services/                # Business logic & data handling
│   │   │   ├── dataset_service.py   # Data cleaning & caching
│   │   │   ├── eda_service.py       # Statistics & visualization data
│   │   │   └── stats_service.py     # Hypothesis testing
│   │   ├── utils/                   # Formatting & math helpers
│   │   ├── config.py                # Environment & path configuration
│   │   └── main.py                  # FastAPI app & CORS middleware
│   ├── models/                      # Serialized scikit-learn artifacts (.joblib)
│   ├── tests/                       # Automated test suite (pytest)
│   ├── inspect_data.py              # CLI diagnostic script
│   ├── main.py                      # Uvicorn entrypoint
│   ├── requirements.txt             # Production Python dependencies
│   ├── pytest.ini                   # Pytest configuration
│   └── .env.example                 # Backend environment template
├── data/
│   ├── raw/
│   │   └── INDIA_RETAIL_DATA.xlsx   # Original dataset (2,534 records)
│   └── processed/
│       └── retail_cleaned.csv       # Preprocessed dataset (23 columns)
├── frontend/
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts               # Type-safe API client
│   │   ├── App.tsx                  # Main React dashboard & views
│   │   ├── index.css                # Figma design styling & Tailwind v4
│   │   └── main.tsx                 # React DOM entrypoint
│   ├── package.json                 # Node dependencies
│   ├── vite.config.ts               # Vite configuration
│   └── .env.example                 # Frontend environment template
└── README.md
```

---

## 3. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System status, documentation URL, and available endpoints |
| `GET` | `/health` | Health check for deployment monitoring |
| `GET` | `/api/dataset/summary` | Dataset metadata, row counts, missing values, and outlier statistics |
| `GET` | `/api/dataset/preview` | Paginated records (`type=raw\|cleaned`, `page`, `page_size`, `search`) |
| `GET` | `/api/eda` | KPIs, monthly trend points, segment breakdown, and correlation matrix |
| `GET` | `/api/statistical-tests` | Mann-Whitney U, Spearman correlation, Chi-Square, and ANOVA results |
| `GET` | `/api/feature-selection` | SelectKBest rankings, correlation ranking, and RF importance (`?k=6`) |
| `GET` | `/api/models/compare` | 4-way model evaluation comparison (MAE, RMSE, $R^2$) and impact deltas |
| `GET` | `/api/models/status` | Model serialization and training status |
| `POST` | `/api/predict` | Make live sales prediction using the trained pipeline |

Interactive Swagger documentation is available at `http://127.0.0.1:8081/docs`.

---

## 4. Local Setup & Running Instructions

### Prerequisites
- Python 3.10+ (Python 3.14 tested and supported)
- Node.js 18+ (Node v24 tested)
- npm or pnpm

### Step 1: Backend Setup & Execution
Open a PowerShell terminal:

```powershell
# Navigate to the backend directory
cd D:\retaillysale\backend

# Activate the existing virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Verify installed dependencies
pip install -r requirements.txt

# Run the standalone diagnostic script
python inspect_data.py

# Run the automated pytest test suite
pytest

# Start the FastAPI server on port 8081
python -m uvicorn app.main:app --host 127.0.0.1 --port 8081 --reload
```

The backend will start and listen at `http://127.0.0.1:8081`.

### Step 2: Frontend Setup & Execution
Open a second PowerShell terminal:

```powershell
# Navigate to the frontend directory
cd D:\retaillysale\frontend

# Ensure the environment variable points to the backend
# (already configured in .env as VITE_API_BASE_URL=http://127.0.0.1:8081)

# Build the production bundle
npm run build

# Start the Vite development server
npm run dev
```

The frontend will start and display a local URL (e.g., `http://localhost:5173` or `http://localhost:8443`). Open this URL in any modern web browser to access the complete interactive interface.

---

## 5. Deployment Guide

### A. Deploying the Backend to Render (Free Web Service)
1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Web Service**.
3. Connect your repository.
4. Configure the following service settings:
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python -c "from app.ml.trainer import ModelTrainer; ModelTrainer.train_and_save(k=6)"`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. In **Environment Variables**, add:
   - `ENVIRONMENT`: `production`
   - `CORS_ORIGINS`: `https://retaillysale.vercel.app,http://localhost:5173`
   - `DATA_RAW_PATH`: `../data/raw/INDIA_RETAIL_DATA.xlsx`
   - `DATA_PROCESSED_PATH`: `../data/processed/retail_cleaned.csv`
   - `MODELS_DIR`: `models`
6. Click **Deploy Web Service**. Once deployed, copy your Render URL (e.g., `https://retaillysale.onrender.com`).

### B. Deploying the Frontend to Vercel
1. In the [Vercel Dashboard](https://vercel.com), click **Add New Project** and import the GitHub repository.
2. In the configuration screen:
   - **Root Directory**: Click edit and select `frontend`.
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Under **Environment Variables**, add:
   - `VITE_API_BASE_URL`: `https://retaillysale.onrender.com` (your deployed Render backend URL)
4. Click **Deploy**. Vercel will build the React bundle and deploy it globally.
5. In the backend on Render, verify that `CORS_ORIGINS` contains your newly generated Vercel production domain.

---

## 6. Academic Contributions & Technical Takeaways

1. **Strict Prevention of Target Leakage**: In business transactions, `Profit` is an ex-post calculated metric ($\text{Profit} = \text{Sales} - \text{Costs}$). Retaining `Profit` as a predictor when estimating `Sales` would create artificial leakage. The pipeline explicitly isolates transaction-time features (`Unit Price`, `QtyOrdered`, `Discount offered`, `Freight Expenses`, `Logistics Mode`).
2. **Train-Only Preprocessing**: Scaling (`StandardScaler`), one-hot encoding (`OneHotEncoder`), and filter-based feature selection (`SelectKBest`) are fit strictly on the 80% training partition, preventing data distribution contamination into the holdout test set.
3. **Statistical Justification**: Hypothesis testing confirms that while promotional discounts do not statistically shift gross transaction sizes ($p = 0.5637$), product unit price demonstrates an exceptionally strong monotonic relationship with sales ($\rho = 0.8400, p < 0.0001$).
4. **Nonlinear Superiority**: Random Forest captures nonlinear interactions between bulk quantity and unit price, achieving an $R^2$ of **0.9401** with feature selection, outperforming the baseline linear model ($R^2 = 0.4533$).
