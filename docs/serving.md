# Lumen Serving API

Lumen provides a robust, production-grade inference server based on FastAPI. It supports async request batching (micro-batching), model alias resolution, and hot-swapping.

## Architecture

The server consists of three main layers:
1. **FastAPI Web Layer**: Handles HTTP requests, image decoding, and response formatting.
2. **Async Batcher**: Collects individual requests and groups them into micro-batches for efficient GPU utilization.
3. **Inference Engine**: Executes the model using `MicroscopyInference`.

## Running the Server

You can start the server using the `lumen serve` CLI command:

```bash
uv run lumen serve --port 8080 --ckpt path/to/model.pt --alias default
```

### Options
- `--port`: Port to listen on (default: 8080).
- `--host`: Host to bind to (default: 0.0.0.0).
- `--alias`: Logical name for the initial model.
- `--ckpt`: Path to the PyTorch checkpoint.

## API Endpoints

### 1. Predict (Single Image)
`POST /v1/predict`

**Request Body:**
```json
{
  "image_b64": "...",
  "model_alias": "default",
  "task_type": "segmentation",
  "confidence": 0.5
}
```

### 2. Predict Batch
`POST /v1/predict/batch`

**Request Body:**
```json
{
  "images": [
    {"image_b64": "...", "task_type": "segmentation"},
    {"image_b64": "...", "task_type": "segmentation"}
  ]
}
```

### 3. Model Management
- `GET /v1/models`: List registered aliases and the current loaded model.
- `POST /v1/models/reload?alias=...`: Hot-swap the active model to a different registered alias.

### 4. Health Checks
- `GET /healthz`: Basic liveness probe.
- `GET /readyz`: Readiness probe (returns 200 only when a model is loaded and batcher is running).

## Configuration

Server-side batching behavior can be configured in the `AsyncBatcher` (currently defaults to `max_batch_size=8` and `max_wait_ms=50`).
