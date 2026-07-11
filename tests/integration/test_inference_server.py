"""Integration tests for the Lumen Inference Server."""

import asyncio
import base64
import io

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from lumen.inference import InferenceConfig, MicroscopyInference
from lumen.serving.app import app
from lumen.serving.registry import registry

# --- Fixtures ---

@pytest.fixture
def sample_image_b64():
    """Create a sample 224x224 grayscale image as base64."""
    img_data = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(img_data)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


@pytest.fixture(autouse=True)
def setup_mock_model(tmp_path):
    """Register a fake model alias so reload endpoints resolve during tests."""
    ckpt_path = tmp_path / "mock_model.pt"
    registry.register_alias("test-model", ckpt_path)
    # Create a dummy file so exists() returns True.
    ckpt_path.touch()
    return ckpt_path


# --- Tests ---

@pytest.mark.asyncio
async def test_health_endpoints():
    """Test healthz and readyz."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_predict_single(sample_image_b64, monkeypatch):
    """Test single image prediction endpoint."""
    # Mock MicroscopyInference to avoid loading real weights/running real inference
    class MockInference:
        def __init__(self, *args, **kwargs):
            self.config = InferenceConfig()
        def load_model(self): pass
        def infer(self, image):
            from lumen.inference import InferenceResult
            # Return dummy predictions matching batch size
            batch_size = image.shape[0] if hasattr(image, "shape") else 1
            return InferenceResult(predictions=np.zeros((batch_size, 224, 224)), batch_size=batch_size)
        def get_model_info(self): return {"status": "mocked"}

    monkeypatch.setattr("lumen.serving.app.MicroscopyInference", MockInference)

    # Load the model
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Trigger reload to use our mocked class
        await ac.post("/v1/models/reload?alias=test-model")

        payload = {
            "image_b64": sample_image_b64,
            "model_alias": "test-model",
            "task_type": "segmentation"
        }
        response = await ac.post("/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["model_alias"] == "test-model"


@pytest.mark.asyncio
async def test_predict_batch(sample_image_b64, monkeypatch):
    """Test batch prediction and micro-batching logic."""
    class MockInference:
        def __init__(self, *args, **kwargs): pass
        def load_model(self): pass
        def infer(self, image):
            from lumen.inference import InferenceResult
            return InferenceResult(predictions=np.zeros((image.shape[0], 224, 224)), batch_size=image.shape[0])
        def get_model_info(self): return {}

    monkeypatch.setattr("lumen.serving.app.MicroscopyInference", MockInference)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/v1/models/reload?alias=test-model")

        payload = {
            "images": [
                {"image_b64": sample_image_b64},
                {"image_b64": sample_image_b64}
            ]
        }
        response = await ac.post("/v1/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_concurrent_stress(sample_image_b64, monkeypatch):
    """Test concurrent requests to ensure batcher handles them."""
    class MockInference:
        def __init__(self, *args, **kwargs): pass
        def load_model(self): pass
        def infer(self, image):
            # Simulate some work
            import time
            time.sleep(0.01)
            from lumen.inference import InferenceResult
            return InferenceResult(predictions=np.zeros((image.shape[0], 224, 224)), batch_size=image.shape[0])
        def get_model_info(self): return {}

    monkeypatch.setattr("lumen.serving.app.MicroscopyInference", MockInference)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.post("/v1/models/reload?alias=test-model")

        # Send 10 simultaneous requests
        tasks = []
        payload = {"image_b64": sample_image_b64}
        for _ in range(10):
            tasks.append(ac.post("/v1/predict", json=payload))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200
            assert "predictions" in resp.json()


@pytest.mark.asyncio
async def test_hot_swap(tmp_path, sample_image_b64, monkeypatch):
    """Test hot-swapping models via the reload endpoint."""
    class MockInference:
        def __init__(self, config):
            self.alias = config.checkpoint_path.stem
        def load_model(self): pass
        def infer(self, image):
            from lumen.inference import InferenceResult
            return InferenceResult(predictions=self.alias, batch_size=1)
        def get_model_info(self): return {}

    monkeypatch.setattr("lumen.serving.app.MicroscopyInference", MockInference)

    m1_path = tmp_path / "model1.pt"
    m1_path.touch()
    m2_path = tmp_path / "model2.pt"
    m2_path.touch()

    registry.register_alias("m1", m1_path)
    registry.register_alias("m2", m2_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Load m1
        await ac.post("/v1/models/reload?alias=m1")
        resp1 = await ac.post("/v1/predict", json={"image_b64": sample_image_b64})
        assert resp1.json()["predictions"] == "model1"

        # Swap to m2
        await ac.post("/v1/models/reload?alias=m2")
        resp2 = await ac.post("/v1/predict", json={"image_b64": sample_image_b64})
        assert resp2.json()["predictions"] == "model2"


def test_real_batcher_infer_path_accepts_hwc_images():
    """Regression: batcher must feed CHW to the CHW-expecting model.

    Runs a real MicroscopyInference (simple encoder, no checkpoint) through the
    AsyncBatcher end to end with NO mocking of infer, so an HWC/CHW mismatch
    surfaces here instead of being hidden behind a mock as in the tests above.
    """
    from lumen.serving.batcher import AsyncBatcher

    cfg = InferenceConfig(
        checkpoint_path=None,
        encoder_name="simple",
        head_name="segmentation",
        task_type="segmentation",
        device="cpu",
        image_size=(64, 64),
        num_classes=3,
    )
    model = MicroscopyInference(cfg)
    model.load_model()
    batcher = AsyncBatcher(model, max_batch_size=4, max_wait_ms=20.0)

    async def _run():
        # HWC uint8 inputs, exactly what decode_image() yields from PNG bytes.
        imgs = [np.random.randint(0, 255, (48, 72), dtype=np.uint8) for _ in range(3)]
        results = await asyncio.gather(*[batcher.predict(im) for im in imgs])
        await batcher.stop()
        return results

    results = asyncio.run(_run())
    assert len(results) == 3
    for pred in results:
        arr = np.asarray(pred)
        assert arr.shape == (64, 64)  # segmentation map at image_size
        assert arr.min() >= 0 and arr.max() < 3  # valid class ids
