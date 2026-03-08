<div align="left">
<img src="https://github.com/user-attachments/assets/41166197-dff9-4859-acea-1e15337e9607" width="300"/>
</div>

Neo is a self-hostable RAG and LLM inference platform. Run it locally, keep your data private, and chat with your files using any model you want.

> [!WARNING]
> Neo is under active development. You may run into bugs, rough edges, or breaking changes along the way.

---

## Getting Started

### Option 1 — Docker (recommended)

**1. Start Ollama**

```bash
ollama serve
```

> [!NOTE]
> Neo uses Ollama to run all LLMs and embedding models. Running Ollama natively (outside Docker) means it can take full advantage of your hardware — Metal on Apple Silicon Macs, CUDA on NVIDIA GPUs, or whatever your device supports. This is especially important on ARM-based machines like M-series MacBooks where Metal acceleration can have compatibility issues inside containers, so running Ollama directly on the host gives you the best performance across all devices.

**2. Start Neo**

```bash
git clone https://github.com/harryfrzz/neo.git
cd neo
docker compose up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

Your chat history, vector indices, and Whisper models are persisted across restarts via Docker volumes.


### Option 2 — Run locally

**1. Start Ollama**

```bash
ollama serve
```

**2. Backend** (Python 3.11+)

```bash
cd backend
pip install -r requirements.txt
fastapi run ExposeAPI.py --host 0.0.0.0 --port 8000
```

System dependencies also needed: `ffmpeg` and `tesseract-ocr`

**3. Frontend** (Node 20+)

```bash
npm install
npm run dev
```
---

## What You Can Do

### Chat
- Create multiple chat sessions and switch between them
- Rename or delete sessions from the sidebar
- Responses stream in real time and can be stopped mid-generation
- Sessions and message history are stored locally in SQLite

### Upload Files and Chat with Them
Each session has its own knowledge base. Upload files and Neo will extract the content, embed it, and use it to answer your questions.

Supported file types:

| Type | Formats |
|---|---|
| Documents | `.pdf`, `.txt`, `.md` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff` |

You can also drag in an entire folder and Neo will ingest everything inside it.


## Under Development

- **Image and video frame-by-frame visual captioning** using BLIP — coming soon
- **Cloud provider support** (OpenAI, Anthropic, Self host in GCP, AWS etc) — coming soon


## Contributing

Contributions are always welcome. Feel free to open an issue or submit a pull request.
