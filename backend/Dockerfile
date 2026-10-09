# Backend runtime image for the AI-Based Urban Traffic Congestion Prediction
# system. One image, one origin: the API serves the static dashboard from
# /app/frontend via FRONTEND_DIR, so a single container publishes the whole
# project.
#
# Build:   docker build -f backend/Dockerfile -t traffic-prediction-api .
# Run:
#     docker run --rm -p 8000:8000 \
#       -e DATABASE_URL=sqlite:////app/data/app.db \
#       -e FRONTEND_DIR=/app/frontend \
#       traffic-prediction-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so rebuilds after source changes reuse the layer.
COPY requirements.txt requirements.txt
COPY backend/requirements.txt backend/requirements.txt
COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The rest of the repository: API package, migrations, ML artifacts, dashboard.
COPY . .

# Non-root user. Render and Docker both execute the image as CMD below.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Apply migrations, then serve. PORT lets Render, Docker and local runs all
# decide the listening port without editing the image.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]