#%%
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16, device_map="cuda")

prompt = "Turn this cat into a dog"
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

# FLUX.2-klein's Qwen3 text encoder can emit float32 hidden states while the
# transformer weights are bfloat16; autocast reconciles the matmul dtypes so the
# context embedder doesn't raise a dtype mismatch.
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    image = pipe(image=input_image, prompt=prompt).images[0]

# Save the image to a file locally. You can also display it in a notebook with `image.show()`.  
image.save("cat_to_dog.png")
# %%
