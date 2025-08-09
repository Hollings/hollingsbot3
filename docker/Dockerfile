FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Make src importable everywhere (bot, celery)
ENV PYTHONPATH=/app/src:/app

# System deps for CairoSVG rendering + basic fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libpangocairo-1.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring the whole repo in (src/, config/, etc.)
COPY . .

# Default for `docker run` (compose can override)
CMD ["python", "-m", "hollingsbot"]
