"""Async request batcher for model inference.

Collects individual requests, micro-batches them based on size or time
windows, and executes them concurrently on the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Coroutine

import numpy as np
import torch

from lumen.data.supervision_bridge import SupervisionBridge
from lumen.inference import InferenceConfig, MicroscopyInference

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request within a batch."""

    image: np.ndarray | torch.Tensor
    future: asyncio.Future[Any]
    task_type: str | None = None
    confidence: float | None = None


class AsyncBatcher:
    """Micro-batching runner for a single model instance."""

    def __init__(
        self,
        model: MicroscopyInference,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self._queue: list[BatchRequest] = []
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"AsyncBatcher started (batch_size={self.max_batch_size}, wait={self.max_wait_ms*1000}ms)")

    async def stop(self) -> None:
        """Stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("AsyncBatcher stopped")

    async def predict(
        self,
        image: np.ndarray | torch.Tensor,
        task_type: str | None = None,
        confidence: float | None = None,
    ) -> Any:
        """Add a request to the queue and await the result."""
        if not self._running:
            await self.start()

        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        request = BatchRequest(
            image=image,
            future=future,
            task_type=task_type,
            confidence=confidence,
        )

        async with self._lock:
            self._queue.append(request)
            # If queue is full, trigger immediate processing if possible
            # (The worker loop will handle this, but we could signal it)

        return await future

    async def _worker_loop(self) -> None:
        """Background loop that processes the queue."""
        while self._running:
            try:
                await asyncio.sleep(0.001)  # Minimal sleep to avoid tight loop
                
                async with self._lock:
                    if not self._queue:
                        continue
                    
                    # Wait for more requests or timeout
                    start_time = time.time()
                    while (
                        len(self._queue) < self.max_batch_size and 
                        (time.time() - start_time) < self.max_wait_ms
                    ):
                        await asyncio.sleep(0.005)
                        if not self._running:
                            break
                    
                    if not self._queue:
                        continue

                    # Extract batch
                    current_batch = self._queue[:self.max_batch_size]
                    self._queue = self._queue[self.max_batch_size:]

                # Process batch outside the lock
                await self._process_batch(current_batch)

            except Exception as e:
                logger.exception(f"Error in AsyncBatcher worker loop: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, requests: list[BatchRequest]) -> None:
        """Normalize, batch, and run inference."""
        if not requests:
            return

        try:
            # 1. Prepare images (grayscale -> 3ch normalization)
            prepared_images = []
            for req in requests:
                # Reuse supervision_bridge logic for normalization
                img = req.image
                # MicroscopyInference.infer handles numpy -> torch and resizing
                # but we want to ensure consistent normalization first if needed.
                # Actually, InferenceServer scope says: 
                # "Reuse the prepare_image_for_supervision percentile-stretch... for grayscale -> 3ch normalisation"
                img_norm = SupervisionBridge.prepare_image(img)
                prepared_images.append(img_norm)

            # 2. Stack into a single tensor
            # Since MicroscopyInference.infer handles single images or batches,
            # we can stack them here.
            batch_tensor = np.stack(prepared_images)

            # 3. Run inference (synchronous call, run in thread to avoid blocking)
            # MicroscopyInference is not async-aware, so we use run_in_executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.model.infer, 
                batch_tensor
            )

            # 4. Scatter results
            predictions = result.predictions
            # result.predictions is (B, ...) or a list/dict of results
            
            for i, req in enumerate(requests):
                if not req.future.done():
                    # Extract i-th result
                    if isinstance(predictions, (list, np.ndarray, torch.Tensor)):
                        req.future.set_result(predictions[i])
                    else:
                        # Fallback if result format is unexpected
                        req.future.set_result(predictions)

        except Exception as e:
            logger.exception(f"Error processing batch: {e}")
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)
