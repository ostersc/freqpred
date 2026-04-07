# Stage 1: build React dashboard
FROM node:22-slim AS ui-builder
WORKDIR /app/ui
COPY freqpred/dashboard/ui/package*.json ./
RUN npm ci --omit=dev
COPY freqpred/dashboard/ui/ .
RUN npm run build

# Stage 2: Python app
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY freqpred/ freqpred/

# Install dependencies (no dev extras)
RUN uv sync --frozen --no-dev

# Copy the built React app into the expected location
COPY --from=ui-builder /app/ui/dist /app/freqpred/dashboard/ui/dist

CMD ["uv", "run", "freqpred", "run"]
