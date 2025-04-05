FROM python:3.12-slim

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

# Create application directories
RUN mkdir -p /app /static

# Copy application files
COPY custom_paperqa_server.py /app/custom_paperqa_server.py
COPY mapping_item.py /app/mapping_item.py

# Create static web folder
WORKDIR /static
# Here you would COPY your static files into the container
COPY ./static_site/ /static/

# Set working directory back to app
WORKDIR /app

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pull Ollama models
RUN ollama serve & \
    until ollama list; do sleep 2; done && \
    ollama pull mistral && \
    ollama pull deepseek-r1:14b && \
    ollama pull nomic-embed-text && \
    pkill -f ollama

# Python dependencies - updated list
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pypdf \
        langchain-community \
        pandas \
        langchain_core \
        tqdm \
        faiss-cpu \
        langchain-ollama \
        openpyxl \
        fastapi \
        uvicorn \
        httpx \
        python-multipart \
        litellm \
        paper-qa \
        PyMuPDF \
        pdfplumber \
        pytest \
        pytest-asyncio

# Create a static file server script
RUN echo '#!/usr/bin/env python3\n\
import http.server\n\
import socketserver\n\
import os\n\
\n\
PORT = 8080\n\
DIRECTORY = "/static"\n\
\n\
class Handler(http.server.SimpleHTTPRequestHandler):\n\
    def __init__(self, *args, **kwargs):\n\
        super().__init__(*args, directory=DIRECTORY, **kwargs)\n\
\n\
def run_server():\n\
    with socketserver.TCPServer(("", PORT), Handler) as httpd:\n\
        print(f"Serving static files at port {PORT} from {DIRECTORY}")\n\
        httpd.serve_forever()\n\
\n\
if __name__ == "__main__":\n\
    run_server()\n\
' > /app/static_server.py && chmod +x /app/static_server.py

# Create start script to run all services
RUN echo '#!/bin/bash\n\
# Start Ollama service\n\
ollama serve & \n\
until ollama list; do sleep 2; done\n\
\n\
# Start PaperQA server\n\
python /app/custom_paperqa_server.py & \n\
\n\
# Start static file server\n\
python /app/static_server.py\n\
' > /start.sh && chmod +x /start.sh

# Expose ports for both servers
EXPOSE 8000 8080

CMD ["/start.sh"]
ENV PATH="/root/.cargo/bin:${PATH}"
