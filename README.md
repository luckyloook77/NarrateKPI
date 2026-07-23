<div align="center">
  <img src="https://img.shields.io/badge/status-MVP-blueviolet?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-3.0.0-teal?style=flat-square&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"/>
  <br/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/Render-deploy-46E3B7?style=flat-square" alt="Render"/>
</div>

<br/>

<p align="center">
  <h1 align="center">📊 NarrateKPI</h1>
  <p align="center"><strong>AI-Driven Agency Report Automation</strong></p>
  <p align="center">
    Transform raw advertising metrics into polished, human-readable client reports<br/>
    with zero-hallucination math, LLM-powered narratives, and a human-in-the-loop review queue.
  </p>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧮 **Zero-Hallucination Math Engine** | Pure Python calculations for CTR, CPA, ROAS, CPC with deterministic anomaly detection (no LLM used for calculations) |
| 🤖 **AI Narrative Generation** | Structured Markdown reports via OpenAI / DeepSeek / Gemini — or dry-run mock mode with no API key |
| 👁️ **Human-in-the-Loop Review** | Dark-mode web UI with split Markdown editor, live preview, KPI metrics bar, and status lifecycle |
| 📧 **One-Click Email Delivery** | Markdown-to-HTML conversion with inline CSS, dispatched via Resend API or logged locally for dry-run |
| 🐳 **Docker-Ready** | Multi-stage Dockerfile for slim production images |
| 🚀 **$0 Cloud Deployment** | Free-tier configs for Render, Koyeb, and any container platform |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Mock Data   │────▶│  Math Engine  │────▶│  LLM Engine  │
│  3 clients   │     │  (Pydantic)   │     │ (OpenAI/Gemini│
│              │     │  Anomaly      │     │  or Dry-Run)  │
│              │     │  Detection    │     │              │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Resend API  │◀────│  FastAPI      │◀────│  Storage     │
│  Email       │     │  Server       │     │  JSON File   │
│  Delivery    │     │  REST API     │     │  DRAFT→SENT  │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  Web UI     │
                    │  /app       │
                    │  Tailwind + │
                    │  Alpine.js  │
                    └─────────────┘
```

### Data Flow

1. **Input**: Client metrics (impressions, clicks, spend, conversions, revenue) — either mock data or custom input
2. **Math Engine**: Derives CTR, CPC, CPA, ROAS. Detects anomalies based on configurable threshold rules (e.g. CPA spike >20% → CRITICAL)
3. **LLM Synthesis**: Takes anomaly report JSON → generates structured Markdown with Executive Summary, Key Insights, and Action Plan
4. **Review Queue**: Human reviews, edits Markdown, approves, and sends via email
5. **Email**: Markdown → styled HTML with inline CSS → Resend API (or dry-run log)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- `pip` (Python package manager)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/narratekpi.git
cd narratekpi
pip install -r requirements.txt
```

### 2. Run Locally

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** (landing page) or **http://localhost:8000/app** (review queue).

### 3. Generate Reports

1. Click **"Generate Weekly Reports"** — runs the pipeline for 3 mock clients
2. Select a client from the sidebar
3. Review the AI-generated Markdown report in the split editor
4. Edit, approve, and send (dry-run if no email API key configured)

---

## 🐳 Docker

### Build & Run

```bash
docker compose up --build
```

Or manually:

```bash
docker build -t narratekpi .
docker run -p 8000:8000 narratekpi
```

### Health Check

The container includes a Docker `HEALTHCHECK` that pings `/api/health` every 30 seconds. Monitor with:

```bash
docker inspect --format='{{json .State.Health}}' narratekpi
```

---

## 📡 API Reference

### `GET /api/health`
Health check endpoint. Returns app version, uptime, store status, and provider configuration.

```json
{
  "status": "ok",
  "version": "3.0.0",
  "uptime_seconds": 42.0,
  "store": { "path": "./reports_store.json", "writable": true },
  "providers": {
    "llm": { "status": "dry_run", "detail": "No API key configured" },
    "email": { "status": "dry_run", "detail": "RESEND_API_KEY not set" }
  }
}
```

### `POST /api/reports/generate-all`
Run the full pipeline for all mock clients. Clears existing reports.

### `GET /api/reports`
List all reports. Optional `?status=DRAFT|IN_REVIEW|APPROVED|SENT` filter.

### `GET /api/reports/{id}`
Get a single report with full Markdown content and raw metrics.

### `PUT /api/reports/{id}`
Update report Markdown content. Body: `{ "markdown_content": "..." }`. Sets status to `IN_REVIEW`.

### `POST /api/reports/{id}/approve`
Approve a report. Only DRAFT and IN_REVIEW reports can be approved.

### `POST /api/reports/{id}/send`
Send an approved report via email. Converts Markdown → HTML → Resend API.

### `POST /api/clients/custom-report`
Create a custom report with arbitrary metrics.

```json
{
  "client_name": "Acme Corp",
  "period": "2026-W30",
  "target_email": "client@acme.com",
  "impressions": 50000,
  "clicks": 2500,
  "spend": 1500.00,
  "conversions": 120,
  "revenue": 8500.00
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | No | — | LLM provider: OpenAI |
| `DEEPSEEK_API_KEY` | No | — | LLM provider: DeepSeek |
| `GEMINI_API_KEY` | No | — | LLM provider: Google Gemini |
| `RESEND_API_KEY` | No | — | Email delivery via Resend |
| `DEFAULT_FROM_EMAIL` | No | `reports@narratekpi.com` | Sender email address |
| `NARRATEKPI_STORE_PATH` | No | `./reports_store.json` | Database file path |

> **Note**: If no LLM API key is set, the app runs in **dry-run mode** — reports are generated from realistic templates. If no email API key is set, emails are logged to `email_output/`.

---

## 🌐 Deployment

### Render (Free Tier)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Push this repo to GitHub
2. In Render Dashboard → **New +** → **Blueprint**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

Or manually:

1. Create a **Web Service** on Render
2. Set **Build Command**: `pip install -r requirements.txt`
3. Set **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Set **Health Check Path**: `/api/health`
5. Deploy!

### Docker on Any Cloud

```bash
docker build -t narratekpi .
docker tag narratekpi ghcr.io/YOUR_USERNAME/narratekpi:latest
docker push ghcr.io/YOUR_USERNAME/narratekpi:latest
```

---

## 🧪 Project Structure

```
narratekpi/
├── server.py            # FastAPI application & REST endpoints
├── storage.py           # JSON file-based report store
├── math_engine.py       # Anomaly detection & metric calculations
├── llm_engine.py        # LLM synthesis (live API or dry-run)
├── prompts.py           # System prompts for LLM
├── email_service.py     # Email dispatch (Resend API or dry-run)
├── schemas.py           # Pydantic v2 data models
├── mock_data.py         # 3 realistic test clients
├── main.py              # CLI entry point
│
├── static/
│   ├── index.html       # Review Queue SPA (Tailwind + Alpine.js)
│   └── landing.html     # Public landing page
│
├── Dockerfile           # Multi-stage Docker build
├── docker-compose.yml   # Local Docker development
├── render.yaml          # Render free-tier deployment
├── requirements.txt     # Python dependencies
├── .dockerignore        # Docker build exclusions
└── README.md            # This file
```

---

## 📊 Anomaly Detection Rules

| Rule | Threshold | Severity |
|---|---|---|
| CPA spike | > +20% | 🔴 CRITICAL |
| CPA drop | > -15% | 🟢 POSITIVE |
| ROAS drop | > -15% | 🔴 CRITICAL |
| ROAS spike | > +15% | 🟢 POSITIVE |
| CTR drop | > -15% | 🟡 WARNING |
| Spend spike + Conversions drop | > +25% spend & drop in conv. | 🔴 CRITICAL |

---

## 🛣️ Roadmap

- [x] **Module 1**: Math Engine & Data Schemas
- [x] **Module 2**: LLM Synthesis Engine
- [x] **Module 3**: HITL Review Queue (FastAPI + Web UI)
- [x] **Module 4**: Custom Ingestion & Email Dispatch
- [x] **Module 5**: Dockerization & $0 Deployment
- [ ] Authentication (Google OAuth / Magic Link)
- [ ] SQLite/Postgres database (replace JSON file)
- [ ] Real ad platform integrations (Google Ads API, Meta Ads)
- [ ] Scheduled report generation (cron / Celery Beat)
- [ ] Multi-language report support

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for agencies that value transparency and speed.
  <br/>
  <a href="https://narratekpi.com">narratekpi.com</a>
</p>
