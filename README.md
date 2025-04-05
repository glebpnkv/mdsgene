
# Running Docker on Google Cloud VM

This guide explains how to set up and run a Dockerized application on a Google Cloud Virtual Machine (VM) instance.

## 🚀 Prerequisites

Before proceeding, ensure that:

- You have a Google Cloud account.
- You have Google Cloud CLI installed
- You have set up a VM instance with Docker installed.
- The VM instance has GPU support enabled (if needed).
- Required ports (`11434`, `8000`) are open in the VM's firewall settings.

## 🐳 Step-by-Step Instructions

### 0. **Install Google Cloud CLI**

From [Using the Google Cloud CLI installer](https://cloud.google.com/sdk/docs/downloads-interactive)

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 1. **Connect to Your VM**

Use SSH to connect to your Google Cloud VM:

```bash
gcloud compute ssh [INSTANCE_NAME] --zone=[ZONE]
```

Replace `[INSTANCE_NAME]` and `[ZONE]` with your VM details.

---

### 2. **Clone or Upload Your Project**

Make sure your Dockerfile and app source code are available on the VM. You can either:
- Use `git clone` to clone your repository, or
- Use `scp` to upload files to the VM.

Example:

```bash
https://github.com/glebpnkv/mdsgene.git
cd mdsgene
```

---

### 3. **Build the Docker Image**

Run the following command to build your Docker image:

```bash
docker build -t my-app .
```

---

### 4. **Run the Docker Container**

To run the Docker container in detached mode with GPU support and expose required ports:

```bash
docker run -d --gpus all -p 11434:11434 -p 8000:8000 --name my-app-container my-app
```

This will:
- Run the container in the background (`-d`)
- Attach all available GPUs (`--gpus all`)
- Map ports `11434` and `8000` from container to host
- Name the container `my-app-container`

---

### 5. **Edit Application Script (Optional)**

You can modify your server script directly in the VM:

```bash
nano custom_paperqa_server.py
```

After editing, you may need to restart the container.

---

### 6. **View Logs**

To see the logs from your running container:

```bash
docker logs my-app-container
```

---

## ✅ Tips

- Use `docker ps` to list running containers.
- Use `docker exec -it my-app-container bash` to access the container’s shell.
- Use `docker stop my-app-container` and `docker rm my-app-container` to stop and remove it.

---

## 🔐 Security Note

Ensure only trusted users can access your VM and avoid exposing sensitive ports to the public without proper authentication.

---

## 📞 Support

For help, visit [Google Cloud Documentation](https://cloud.google.com/docs) or open an issue in your repository.

---

## 🧠 Running the Excel Mapping Application (Locally)

This application extracts structured patient data from a PDF using a custom RAG engine and formats it using a local Ollama instance with the `mistral` model.

### 🔧 Prerequisites

- Python 3.8+
- Docker is **not required** for this part
- Ollama with `mistral` model installed and running locally:
  
```bash
ollama run mistral
```

- Install required Python packages. You may use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Make sure you have the following custom modules available:
- `mapping_item.py` (contains `MappingItem` and `QuestionInfo`)
- `rag_query_engine.py` (your RAG logic)
- `excel_writer.py` (writes to Excel)

### ▶️ Run the App

After setting up the environment and ensuring Ollama is running, start the application:

```bash
python excel_mapping_app.py
```

Make sure to adjust the path to the PDF in the script (`PDF_FILEPATH`) if needed.

---

## 📂 Output

The application generates or appends to:

```
./Patient_Data_Appended_Python.xlsx
```

This file will contain one row per identified patient, with formatted values based on LLM output.

---

## 🛠 Troubleshooting

- Ensure `ollama run mistral` is running **before** launching the app.
- Check Docker logs if you are running a RAG service in a container.
- Logs and outputs will appear in the terminal.


---

## 🌐 Environment Variable

Make sure the following environment variable is correctly set before running the application:

```bash
export PAPERQA_SERVER_URL="http://your-paperqa-server:8000/ai_prompt"
```

If not set, it will default to:

```
http://34.147.75.119:8000/ai_prompt
```

You can also set this inline when running the script:

```bash
PAPERQA_SERVER_URL=http://your-paperqa-server:8000/ai_prompt 
python excel_mapping_app.py
```

This variable points to an external service used for identifying patient IDs from documents. The URL must be reachable from your local environment.
