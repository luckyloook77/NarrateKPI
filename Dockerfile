# ─────────────────────────────────────────────────────────────────────
#  NarrateKPI — Dockerfile
#  Base: python:3.11-slim
#  Exposes: 8000
#  Entrypoint: uvicorn server:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Layer 1: Install build deps & compile dependencies ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Layer 2: Runtime image ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy only the installed packages from builder (skip build tools)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY *.py ./
COPY static/ ./static/

# Create directory for store persistence
RUN mkdir -p /data && chmod 755 /data

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
