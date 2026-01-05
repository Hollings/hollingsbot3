FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# Make src importable everywhere (bot, celery)
ENV PYTHONPATH=/app/src:/app

# System deps for CairoSVG rendering + basic fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 libpangocairo-1.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI and Compose plugin so the bot can control Docker when the socket is mounted
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && . /etc/os-release \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$ID $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Claude Code CLI for agentic search
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for Claude CLI (it refuses --dangerously-skip-permissions as root)
RUN useradd -m -s /bin/bash claude \
    && mkdir -p /home/claude/.claude \
    && chown -R claude:claude /home/claude

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring the whole repo in (src/, config/, etc.)
COPY . .

# Default for `docker run` (compose can override)
CMD ["python", "-m", "hollingsbot"]
