# Data Agent — AI-Powered Data Analysis

A full-stack intelligent data analysis application built with **LangChain + LangGraph** and **React + TypeScript**. Upload a CSV, ask questions in natural language, and get instant analysis — statistics, aggregations, and baseline machine learning — all executed in a secure sandbox.

## Features

### Data Management
- **CSV upload** — upload files directly in the chat input area
- **Data preview** — first 10 rows with column type inference (`numerical` / `categorical`)
- **Multi-layer semantics** — preserves `raw_df`; generates `analysis_df` on demand (exposed as `working_df`)
- **Schema profiling** — automatic `schema_profile` with column semantics, usability flags, and warnings
- **Transparent preprocessing** — logged step-by-step with retained/excluded columns and warnings
- **Model prep planning** — on-demand `model_prep_plan` for target candidates and feature availability

### AI Analysis
- **Streaming chat** — real-time Server-Sent Events (SSE) with DeepSeek
- **Secure code execution** — Python runs in a sandbox with AST validation, timeout protection, and read-only DataFrame
- **Helper APIs** — `data.*`, `stats.*`, `profile.*`, `ml.*` for common operations without writing pandas code
- **Statistical analysis** — descriptive stats, grouping, correlation, t-test, chi-square, ANOVA
- **Baseline ML** — logistic regression & linear regression with metrics and feature importance
- **Dynamic context** — agent system prompt auto-updates with the active dataset schema

## Project Structure

```
data_agent_lite/
├── backend/
│   ├── src/
│   │   ├── agent.py              # LangGraph agent definition & system prompt
│   │   ├── server.py             # FastAPI server
│   │   ├── data_manager.py       # Dataset lifecycle & repository
│   │   ├── tools.py              # Agent tools (python_inter, stats_execute, ml_execute)
│   │   ├── safe_executor.py      # AST validator, sandbox, read-only proxies
│   │   ├── self_correction.py    # Error classification & repair prompts
│   │   ├── routing_executor.py   # Streaming execution & tool routing
│   │   ├── chat_service.py       # Chat pipeline & exception classification
│   │   ├── stats_service.py      # Statistical helper APIs
│   │   ├── ml_helpers.py         # Baseline ML helper APIs
│   │   ├── profile_service.py    # Schema profiling helpers
│   │   ├── preprocessing.py      # Data preprocessing & model prep
│   │   └── settings.py           # Environment-based configuration
│   ├── static/                   # Static files
│   ├── temp_data/                # Temporary uploaded CSVs
│   └── pyproject.toml            # Python dependencies (uv)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── chat/             # Chat hooks, message views, SSE parsing
│   │   │   └── ui/               # UI component library
│   │   ├── config/
│   │   │   └── api.ts            # API base URL configuration
│   │   └── App.tsx
│   └── package.json
│
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [DeepSeek API Key](https://www.deepseek.com/)

### Backend Setup

```bash
cd backend

# Install dependencies (requires uv: https://docs.astral.sh/uv/)
uv sync

# Create .env file
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Start Services

**Backend** (port 8002):
```bash
cd backend
uv run python -m src.server
```

**Frontend** (port 5173):
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

## Configuration

### Backend (`.env`)

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
# Optional: CORS whitelist for production (comma-separated)
CORS_ALLOW_ORIGINS=https://your-frontend.example.com
```

### Frontend (`src/config/api.ts`)

```typescript
const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8002';
```

In production, set `VITE_API_BASE_URL` environment variable or use the `/api` proxy path.

## Usage

### 1. Upload Data
Click the upload button in the chat input to select a CSV file. A data preview card with the first 10 rows and column types will appear.

### 2. Ask Questions

Natural language examples:
- *"Describe the numeric columns"*
- *"Group by Contract and compute Churn rate, sorted descending"*
- *"Calculate total MonthlyCharges by gender and Contract using python_inter"*
- *"Train a logistic regression to predict Churn and show accuracy"*
- *"Run a t-test on MonthlyCharges between Churn=Yes and Churn=No"*

### Tool Reference

| Tool | Description |
|---|---|
| `python_inter(py_code)` | Execute pandas/numpy code in a secure sandbox. Available: `df`, `pd`, `np`, `data.*`, `stats.*`, `profile.*`, `ml.*` |
| `stats_execute(action, ...)` | Statistical analysis shortcut (describe, group summary, t-test, chi-square, etc.) |
| `ml_execute(action, ...)` | ML shortcut (train logistic/linear regression, metrics, feature importance) |

### Helper APIs (available inside `python_inter`)

| Group | Methods |
|---|---|
| `data` | `.head()`, `.describe()`, `.value_counts()`, `.correlation()`, `.group_mean()`, `.filter_equals()`, `.select()` |
| `stats` | `.describe_numeric()`, `.group_summary()`, `.t_test()`, `.chi_square()`, `.anova()` |
| `profile` | `.schema()`, `.analysis_preprocess()`, `.model_prep_plan()` |
| `ml` | `.logistic_fit()`, `.linear_regression_fit()`, `.metrics()`, `.feature_importance()` |

### Current Limitations

- No arbitrary `import` — only `pd` and `np` are pre-injected
- No plotting/charting — results are returned as tables and text
- Baseline ML limited to logistic & linear regression
- No AutoML, hyperparameter search, or model persistence
- Statistical tests are for exploratory analysis; small samples trigger warnings

## Deployment

> **Security note**: This version uses "restricted execution + AST validation + timeout protection" as a practical defense model. It significantly reduces risk but is **not** OS-level sandbox isolation. In production, deploy behind a container with minimal privileges and network isolation.

### Railway (Backend)
Set `DEEPSEEK_API_KEY` as an environment variable. The server listens on `$PORT` (default 8080).

### Vercel (Frontend)
Set `VITE_API_BASE_URL` to your Railway backend URL during build.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM | DeepSeek (`deepseek-chat`) |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite |
| Data | Pandas + NumPy + SciPy |
| ML | scikit-learn (logistic/linear regression) |
| Package Manager | uv (Python), npm (Node.js) |
