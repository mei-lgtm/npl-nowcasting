# NPL Nowcasting Intelligence System

AI-powered **Non-Performing Loan (NPL) nowcasting** with news/media sentiment analytics.

The system estimates the **current / latest NPL condition** before official statistics are released, while strictly distinguishing:

| Type | Meaning |
|------|---------|
| **Actual NPL** | Officially released historical NPL |
| **Nowcast NPL** | Model estimate for the latest *unreleased* period(s) |
| **Forecast NPL** | Projection for *future* periods |

> **DEMO MODE** ships with clearly labeled **synthetic data**. It must never be presented as official Indonesian statistics.

---

## Architecture

```text
Frontend (Next.js)
   ↓
API Layer (FastAPI)
   ↓
Data Pipeline + Connectors → Standardized Schema
   ↓
Vintage Engine (publication-date / news timing)
   ↓
NLP Sentiment Engine → News Stress Index
   ↓
Feature Engineering + Variable Selection
   ↓
Nowcasting Engine (econometric / MIDAS / ML / hybrid)
   ↓
Temporal Backtest + Auto Model Selection
   ↓
Dashboard / AI Analyst / Scenarios
```

### Stack

- **Frontend:** Next.js 14, React, Recharts
- **Backend:** Python, FastAPI, pandas, statsmodels, scikit-learn, XGBoost, LightGBM
- **Demo storage:** Parquet files (swap to PostgreSQL in production via connectors)

---

## Quick start

### Backend + built-in UI (no Node required)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-min.txt
uvicorn main:app --reload --port 8000
```

Open the intelligence UI: **http://127.0.0.1:8000**  
API docs: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/health

The UI matches the editorial paper design in `npl-nowcasting.html` (Source Serif 4 + Inter, Bahasa Indonesia, full workflow pages). It runs entirely in the browser with clearly labeled synthetic demo data.

### Optional Next.js frontend

Requires Node.js 18+.

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:3000 (proxies API to port 8000)

### Optional: XGBoost / LightGBM on macOS

```bash
brew install libomp
pip install xgboost lightgbm
```

Without OpenMP, those models automatically fall back to scikit-learn gradient boosting.

---

## Deploy (GitHub + Vercel)

Repo target: **https://github.com/mei-lgtm/npl-nowcasting**  
Vercel team: **mei-lgtms-projects** (`vercel.com/mei-lgtm`)

### 1. Push to GitHub

```bash
git init -b main
git add -A && git commit -m "Initial commit: NPL nowcasting system"
gh auth login   # authorize in browser
gh repo create mei-lgtm/npl-nowcasting --public --source=. --remote=origin --push
```

### 2. Deploy on Vercel

```bash
npm i -g vercel
vercel login    # authorize in browser
vercel link --yes --scope mei-lgtms-projects
vercel --prod
```

Or in the Vercel dashboard: **Add New Project** → Import `mei-lgtm/npl-nowcasting` → Deploy.  
Framework preset can stay **Other**; `vercel.json` routes all traffic through the FastAPI Mangum handler (`api/index.py`).

> Serverless has a **60s** request limit and a lean dependency set (`requirements.txt` at repo root). For long ML jobs locally, keep using `uvicorn` on port 8000.

---

## Primary user flow

1. Open **Run Workflow** (or Dashboard auto-runs demo config)
2. Select target NPL, sentiment model, variable mode, model selection mode
3. Run pipeline → backtest → rank models → current nowcast
4. Inspect drivers, news articles, sector stress, scenarios, track record

---

## Critical econometric rules enforced

- No look-ahead: `publication_date <= as_of` and `published_at <= nowcast_timestamp`
- Temporal cross-validation only (expanding / rolling) — never shuffle
- Auto model selection ranks by **out-of-sample RMSE / MAE**
- Deep learning models are skipped / fall back when the sample is too small
- Feature importance is labeled as **association**, not causation
- Demo / synthetic outputs are always flagged

---

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `NPL_APP_MODE` | `demo` | `demo` or `production` |
| `NPL_OPENAI_API_KEY` | — | Optional LLM sentiment |
| `NPL_DATABASE_URL` | sqlite | Production Postgres URL |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | Frontend API base |

---

## Modules to extend

| Concern | Location |
|---------|----------|
| Data connectors (OJK, BI, BPS, RSS) | `backend/app/data/connectors/` |
| Sentiment models | `backend/app/engines/sentiment/` |
| Nowcasting models | `backend/app/engines/nowcasting/models.py` |
| Vintage / real-time logic | `backend/app/engines/vintage/` |
| Feature engineering | `backend/app/engines/features/` |
| UI pages | `frontend/app/` |

---

## License

Internal / research prototype. Synthetic demo data is fabricated.
