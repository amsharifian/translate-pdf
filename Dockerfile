# ── Stage 1: Build dependencies ─────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install only the CPU version of torch (much smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and assets
COPY translator/ translator/
COPY scripts/ scripts/
COPY assets/ assets/
COPY app.py pyproject.toml requirements.txt config.yml ./

# Install the translator package in editable mode
RUN pip install --no-cache-dir --no-deps -e .

# Create directories for jobs and data persistence
RUN mkdir -p jobs translated

# Streamlit config: disable telemetry, set server options
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8501/_stcore/health')" || exit 1

# Start worker + streamlit
CMD ["python", "scripts/run_stormlight.py"]
