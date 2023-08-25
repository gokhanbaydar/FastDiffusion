from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from core import imgutils

import torch
import time


class FastDiffusionGenerator:
    def __init__(self):
        self.models_loaded = False
        self.proc_start_time = None
        self.proc_elapsed_time = None

    def load_models(self):
        self.proc_start_time = time.time()
        self.controlnet_conditioning_scale = 0.5  # recommended for good generalization
        self.controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
        )
        self.pipe.enable_model_cpu_offload()
        self.proc_elapsed_time = time.time() - self.proc_start_time
        self.proc_start_time = None
        self.models_loaded = True

    def run(
        self,
        control_image: str,
        prompt: str,
        negative_prompt: str = "low quality, bad quality, sketches",
        control_schema: str = "canny",
    ):
        self.proc_start_time = time.time()
        image = imgutils.canny_image(control_image)
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
        ).images
        images[0].save(f"{prompt}.png")
        self.proc_elapsed_time = time.time() - self.proc_start_time
        self.proc_start_time = None
        return images[0]
