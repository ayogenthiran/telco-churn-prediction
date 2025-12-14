# 1. Use the official lightweight Python base image
# python:3.11-slim is optimal for ML serving: small (~45MB base) with good compatibility
# Most ML packages (XGBoost, LightGBM, NumPy, Pandas) have pre-built wheels
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency files first (for Docker layer caching)
COPY pyproject.toml .

# 4. Install system dependencies and Python packages
# Build dependencies are removed after pip install to keep image size small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --upgrade pip \
    && pip install --no-cache-dir . \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
# .dockerignore ensures only necessary files are copied
COPY . .

# 7. Set environment variables
# PYTHONUNBUFFERED: ensures logs are shown in real-time (no buffering)
# PYTHONPATH: makes "serving" and "app" importable without the "src." prefix
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Run the FastAPI app using uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]