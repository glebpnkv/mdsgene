FROM python:3.13-slim

# Install dependencies, including procps (for pkill)
RUN apt-get update --allow-insecure-repositories -o Acquire::AllowInsecureRepositories=true && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        build-essential \
        ca-certificates \
        procps && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

ENV PATH="/root/.cargo/bin:${PATH}"

ENV PATH="${PATH}:/root/.cargo/bin"

ENV CARGO_HOME=/root/.cargo

COPY custom_paperqa_server.py /app/custom_paperqa_server.py
WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pull Ollama models
RUN ollama serve & \
    until ollama list; do sleep 2; done && \
    ollama pull deepseek-r1:14b && \
    ollama pull nomic-embed-text && \
    pkill -f ollama

# Make start script
RUN echo '#!/bin/bash\n\
ollama serve & \
until ollama list; do sleep 2; done \
&& python custom_paperqa_server.py' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]

ENV PATH="/root/.cargo/bin:${PATH}"

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uvicorn fastapi httpx python-multipart paper-qa PyMuPDF pdfplumber

EXPOSE 8000
