"""FastAPI application for Lumen inference serving.

Provides standard REST endpoints for prediction, model management,
and health monitoring.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from PIL import Image
from pydantic import BaseModel, Field

from lumen.inference import InferenceConfig, MicroscopyInference
from lumen.serving.batcher import AsyncBatcher
from lumen.serving.registry import registry

logger = logging.getLogger(__name__)


# --- Models ---

class PredictRequest(BaseModel):
    """Request for single image prediction."""
    image_url: str | None = None
    image_b64: str | None = None
    model_alias: str = "default"
    task_type: Literal["classification", "segmentation", "detection"] = "segmentation"
    confidence: float = 0.5


class BatchPredictRequest(BaseModel):
    """Request for multiple images prediction."""
    images: list[PredictRequest]


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    id: str
    info: dict[str, Any]


# --- Global State ---

class ServiceState:
    """Container for active models and batchers."""
    
    def __init__(self) -> None:
        self.batcher: AsyncBatcher | None = None
        self.current_alias: str | None = None

    async def load_model(self, alias: str) -> None:
        """Load or reload a model by alias."""
        try:
            ckpt_path = registry.resolve_alias(alias)
            logger.info(f"Loading model from alias {alias!r} -> {ckpt_path}")
            
            # Create inference engine
            config = InferenceConfig(
                checkpoint_path=ckpt_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            inference = MicroscopyInference(config)
            inference.load_model()
            
            # Stop old batcher if exists
            if self.batcher:
                await self.batcher.stop()
            
            # Create new batcher
            self.batcher = AsyncBatcher(inference)
            await self.batcher.start()
            self.current_alias = alias
            
            logger.info(f"Model {alias!r} loaded and ready for serving")
        except Exception as e:
            logger.error(f"Failed to load model {alias!r}: {e}")
            raise


state = ServiceState()


# --- Utils ---

def decode_image(request: PredictRequest) -> np.ndarray:
    """Decode image from URL or Base64."""
    if request.image_b64:
        try:
            content = base64.b64decode(request.image_b64)
            img = Image.open(io.BytesIO(content))
            return np.array(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    
    if request.image_url:
        try:
            import httpx
            # Synchronous fetch for simplicity in this utility, 
            # but usually should be async in the endpoint.
            # For brevity, let's assume b64 or local paths for now 
            # as per roboflow-inference parity.
            raise HTTPException(status_code=501, detail="image_url not yet implemented")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
            
    raise HTTPException(status_code=400, detail="Either image_url or image_b64 must be provided")


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start up and shut down logic."""
    # Attempt to load default model if registered
    aliases = registry.list_aliases()
    if aliases:
        try:
            await state.load_model(aliases[0])
        except Exception:
            logger.warning("Could not load default model on startup")
    yield
    # Shutdown
    if state.batcher:
        await state.batcher.stop()


app = FastAPI(
    title="Lumen Inference Server",
    description="Production-grade microscopy inference API",
    version="0.1.0",
    lifespan=lifespan
)


# --- Endpoints ---

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Basic health check."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    """Readiness check."""
    if state.batcher and state.batcher._running:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available model aliases and current loaded model."""
    return {
        "aliases": registry.list_aliases(),
        "current": state.current_alias,
        "info": state.batcher.model.get_model_info() if state.batcher else None
    }


@app.post("/v1/models/reload")
async def reload_model(alias: str = Query(..., description="Alias to load")) -> dict[str, str]:
    """Hot-swap the current model."""
    try:
        await state.load_model(alias)
        return {"status": "ok", "message": f"Model {alias} loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/predict")
async def predict(request: PredictRequest) -> dict[str, Any]:
    """Single image prediction with async micro-batching."""
    if not state.batcher:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    image_arr = decode_image(request)
    
    # Run through batcher
    prediction = await state.batcher.predict(
        image=image_arr,
        task_type=request.task_type,
        confidence=request.confidence
    )
    
    # Format result (supervision-compatible)
    # The batcher returns what MicroscopyInference.infer returns (predictions field)
    # which is usually a numpy array.
    return {
        "predictions": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
        "model_alias": state.current_alias,
        "task_type": request.task_type
    }


@app.post("/v1/predict/batch")
async def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    """Batch prediction with async micro-batching."""
    if not state.batcher:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    # Process all in parallel
    tasks = []
    for req in request.images:
        image_arr = decode_image(req)
        tasks.append(state.batcher.predict(
            image=image_arr,
            task_type=req.task_type,
            confidence=req.confidence
        ))
    
    results = await asyncio.gather(*tasks)
    
    formatted_results = []
    for res in results:
        formatted_results.append(
            res.tolist() if hasattr(res, "tolist") else res
        )
        
    return {
        "results": formatted_results,
        "model_alias": state.current_alias
    }

# Ensure torch is available globally for the lifespan/state
import torch
