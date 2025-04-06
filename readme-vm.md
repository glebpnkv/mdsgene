# PaperQA Docker Setup Guide

This guide explains how to set up and run the PaperQA application with both the API server and static web interface.

## Prerequisites

- Docker installed on your Linux machine
- GPU support (for optimal performance)
- Basic familiarity with terminal commands

## Project Structure

Your project should be organized as follows:

```
project_root/
├── Dockerfile
├── custom_paperqa_server.py
└── static_site/
    ├── index.html
    └── [other static files and subdirectories]
```

## Setup Instructions

### 1. Configure Docker to use Nvidia driver

```bash
# Installing nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Prepare Your Static Files

Place all your static web files (HTML, CSS, JavaScript, images, etc.) in the `static_site` directory. The entire directory structure will be copied to the container.

```bash
mkdir -p static_site
# Copy your static files here
```

### 3. Build the Docker Image

From your project root directory, build the Docker image:

```bash
docker build -t my-app .
```

This process may take some time as it:
- Pulls the base Python image
- Installs system dependencies
- Sets up Rust
- Installs Ollama
- Downloads the required LLM models (mistral, deepseek-r1:14b, nomic-embed-text)
- Installs Python dependencies

### 4. Run the Container

Run the container with GPU support and expose all necessary ports:

```bash
docker run -d --gpus all \
  -e OLLAMA_HOST=0.0.0.0 \
  -p 11434:11434 \
  -p 8000:8000 \
  -p 8080:8080 \
  --name my-app-container \
  my-app
```

Environment variables:
- `OLLAMA_HOST=0.0.0.0` - Allows forwarding Ollama requests 

Port mappings:
- `11434:11434` - Ollama API
- `8000:8000` - PaperQA FastAPI server
- `8080:8080` - Static web server

### 5. Verify Services

Check if all services are running:

```bash
# Check if the container is running
docker ps

# Check the logs
docker logs my-app-container

# Test the FastAPI endpoint
curl http://localhost:8000

# Test the static web server
curl http://localhost:8080
```

You should be able to access:
- The PaperQA API at http://localhost:8000
- The static web interface at http://localhost:8080

## Troubleshooting

If you encounter issues:

1. **Container fails to start:**
   ```bash
   docker logs my-app-container
   ```

2. **Services not responding:**
   Connect to the container and check running processes:
   ```bash
   docker exec -it my-app-container /bin/bash
   ps aux | grep python
   netstat -tulpn | grep LISTEN
   ```

3. **Port conflicts:**
   If ports are already in use, change the host ports in the docker run command.

4. **Manual service start:**
   Inside the container:
   ```bash
   # Start Ollama
   ollama serve &
   
   # Start PaperQA server
   python /app/custom_paperqa_server.py &
   
   # Start static file server
   python /app/static_server.py
   ```

## Stopping and Removing

```bash
# Stop the container
docker stop my-app-container

# Remove the container
docker rm my-app-container

# Remove the image (optional)
docker rmi my-app
```

## Container Management

- Restart container: `docker restart my-app-container`
- Stop container: `docker stop my-app-container`
- View logs: `docker logs my-app-container`
