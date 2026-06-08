"""Example: Running the Lumen Inference Server.

This example shows how to start the FastAPI inference server and interact
with it via curl or the httpx client.

To run this example:
    uv run lumen serve --port 8080 --ckpt /path/to/checkpoint.pt --alias my-model
"""

import base64
import io
import json
import numpy as np
from PIL import Image

def generate_sample_request():
    """Generate a sample base64 request for testing."""
    # Create a dummy image
    img_data = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(img_data)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    request_body = {
        "image_b64": img_b64,
        "model_alias": "my-model",
        "task_type": "segmentation",
        "confidence": 0.5
    }
    
    print("Sample curl command:")
    print("--------------------")
    print("curl -X POST http://localhost:8080/v1/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print(f"     -d '{json.dumps(request_body)[:100]}... (truncated)'")
    
    return request_body

if __name__ == "__main__":
    print("Lumen Inference Server Example")
    print("==============================")
    print("\n1. Start the server:")
    print("   uv run lumen serve --port 8080 --ckpt path/to/your/model.pt --alias default")
    
    print("\n2. Example API Call:")
    generate_sample_request()
    
    print("\n3. Batch Prediction:")
    print("   curl -X POST http://localhost:8080/v1/predict/batch ...")
    
    print("\n4. Hot-swap Model:")
    print("   curl -X POST 'http://localhost:8080/v1/models/reload?alias=new-alias'")
