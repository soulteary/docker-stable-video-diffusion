# improve & refs https://huggingface.co/spaces/multimodalart/stable-video-diffusion

import torch
import os
from glob import glob
from typing import Optional, Tuple
import random

import gradio as gr
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image


max_64_bit_int = 2 ** 63 - 1
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "/app/models/stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)

# According to your actual needs
#
# pipe.enable_model_cpu_offload()
# pipe.unet.enable_forward_chunking()

def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt",
    cond_aug: float = 0.02,
    decoding_t: int = 3,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = output_folder,
):

    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)

    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    frames = pipe(
        image,
        decode_chunk_size=decoding_t,
        generator=torch.manual_seed(seed),
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=0.1,
        num_frames=25,
    ).frames[0]

    export_to_video(frames, video_path, fps=fps_id)

    return video_path, seed


def resize_image(image: Image, output_size: Tuple[int, int] =(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    # set correct image mode
    if cropped_image.mode == "RGBA":
        cropped_image = cropped_image.convert("RGB")

    return cropped_image


def generate(image, seed, randomize_seed, motion_bucket_id, fps_id):
    img = resize_image(image, output_size=(1024, 576))
    video, seed = sample(img, seed, randomize_seed, motion_bucket_id, fps_id)
    return video, seed


app = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(label="Upload your image", type="pil"),
        gr.Slider(
            label="Seed",
            value=42,
            randomize=True,
            minimum=0,
            maximum=max_64_bit_int,
            step=1,
        ),
        gr.Checkbox(label="Randomize seed", value=True),
        gr.Slider(
            label="Motion bucket id",
            info="Controls how much motion to add/remove from the image",
            value=127,
            minimum=1,
            maximum=255,
        ),
        gr.Slider(
            label="Frames per second",
            info="The length of your video in seconds will be 25/fps",
            value=6,
            minimum=5,
            maximum=30,
        ),
    ],
    outputs=[
        gr.PlayableVideo(label="Generated video"),
        gr.Textbox(label="Seed", type="text"),
    ],
)

if __name__ == "__main__":
    app.queue(max_size=2)
    app.launch(share=False, server_name="0.0.0.0", ssl_verify=False)
