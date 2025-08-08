FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# So Python can see your src/ package everywhere (bot, celery)
ENV PYTHONPATH=/app/src:/app

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# bring the whole repo in (src/, config/, etc.)
COPY . .

# Default entrypoint unused by compose (compose sets command),
# but keep a sane default for `docker run`:
CMD ["python", "-m", "hollingsbot"]
