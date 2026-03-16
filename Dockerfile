FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml .
COPY freqpred/ freqpred/

# Install dependencies (no dev extras)
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "freqpred", "run"]
