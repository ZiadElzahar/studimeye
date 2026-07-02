# AI Support Agent API

A production-ready, multimodal AI microservice designed to process customer support tickets. It accepts text, images, and audio, routing them through specialized AI tools (OCR, Speech-to-Text, LLM Classification) to extract structured metadata, priorities, and actionable recommendations.

Built with **FastAPI** and **PyTorch**, designed to be consumed by a Node.js (or any HTTP) backend.

## Features
- **Multimodal Routing**: Automatically processes text, extracts text from images (DeepSeek-OCR), and transcribes audio (Whisper) before classification.
- **Generative Triaging**: Uses Microsoft's Phi-3-mini to classify tickets into a strict taxonomy and generate immediate support suggestions.
- **Core Inquiry Extraction**: Uses Qwen2.5 to summarize the root cause of the ticket.
- **Production Ready**: Singleton model loading, strict Pydantic validation, global exception handling, and temporary file cleanup.

## Project Structure
```text
project_root/
├── agent/                     # Core business logic and routing
│   ├── orchestrator.py        # Main entry point, handles execution flow
│   ├── router.py              # Routes traffic based on modality presence
│   ├── schemas.py             # Internal Pydantic models (Ticket, AgentResult)
│   ├── services.py            # Tool execution wrappers (Timeouts, Errors)
│   └── exceptions.py          # Custom exception classes
├── tools/                     # Independent AI model wrappers
│   ├── text_tool.py           # Phi-3-mini-4k-instruct (Classification)
│   ├── image_tool.py          # DeepSeek-OCR (Image extraction)
│   ├── speech_tool.py         # Whisper-large-v3 (Audio transcription)
│   └── recommendation_tool.py # Qwen2.5-1.5B-Instruct (Summarization)
├── api/                       # FastAPI REST layer
│   ├── main.py                # App entry point, lifespan, exception handlers
│   ├── routes.py              # POST /predict and GET /health endpoints
│   └── schemas.py             # API request/response models
├── requirements.txt
└── README.md
```

## Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with at least **8GB VRAM** (Recommended for running all models concurrently. A 12GB+ GPU is ideal).
- CUDA Toolkit installed (for PyTorch GPU support).

## Installation & Setup

1. **Clone the repository and navigate to the folder:**
   ```bash
   cd project_root
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory. If you are running outside of Kaggle/Colab, you must provide a HuggingFace token for the Whisper API client.
   ```env
   HF_TOKEN=hf_your_token_here
   ```

## Running the API

Start the FastAPI server using Uvicorn. We use `--workers 1` because loading multiple instances of large AI models on a single GPU will cause Out-Of-Memory (OOM) errors.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The API will be available at `http://localhost:8000`. Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

## API Endpoints

### `POST /predict`
Processes a support ticket. Accepts `multipart/form-data`.

**Form Fields:**
- `text` (string, optional): The text content of the ticket.
- `image` (file, optional): Image attachment (PNG, JPG).
- `audio` (file, optional): Audio attachment (OGG, WAV, MP3).
- `metadata` (string, optional): JSON string containing additional metadata (e.g., `{"ticket_id": "T-123"}`).

*Note: At least one of `text`, `image`, or `audio` must be provided.*

**Response Schema:**
```json
{
  "category": "security",
  "priority": "P2",
  "department": "Security",
  "recommendations": [
    "Dispatch nearest security patrol to Gate D immediately.",
    "Review CCTV footage for suspicious activities."
  ],
  "confidence": 0.97
}
```

### `GET /health`
Health check endpoint for load balancers.
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Node.js Integration Example

Here is how your Node.js backend should call the API:

```javascript
const FormData = require('form-data');
const axios = require('axios');
const fs = require('fs');

async function processTicket() {
    const form = new FormData();
    form.append('text', 'Someone stole my backpack at Gate D!');
    form.append('image', fs.createReadStream('path/to/image.png'), 'image.png');
    form.append('metadata', JSON.stringify({ ticket_id: 'T-12345' }));

    try {
        const response = await axios.post('http://localhost:8000/predict', form, {
            headers: form.getHeaders(),
            // Optional: timeout configuration
            timeout: 180000 
        });
        console.log(response.data);
    } catch (error) {
        console.error('Error processing ticket:', error.response?.data || error.message);
    }
}
```

## Production Notes & Audit Recommendations

1. **VRAM Management**: The `lifespan` startup event loads all models into VRAM simultaneously. If you experience CUDA OOM errors on lower-end GPUs (e.g., 4GB-6GB), modify the `Orchestrator` to dynamically load/unload tools using `del model` and `torch.cuda.empty_cache()`.
2. **Thread Safety**: PyTorch models on a single GPU are not thread-safe for concurrent forward passes. Ensure your load balancer restricts concurrent requests to 1 per GPU container, or implement an `asyncio.Semaphore(1)` in the Orchestrator.
3. **File Cleanup**: The `/predict` endpoint saves uploaded files to a temporary directory and deletes them after processing to prevent disk pollution.
```