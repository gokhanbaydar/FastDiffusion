from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from core import imgutils

import torch
import time


class FastDiffusionGenerator:
    def __init__(self):
        self.models_loaded = False
        self.proc_start_time = None
        self.proc_elapsed_time = None
        self.control_schema = None
        self.controlnet_conditioning_scale = 0.5  # recommended for good generalization
        self.controlnet = None
        self.pipe = None

    def load_models(self):
        self.proc_start_time = time.time()
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        if self.controlnet:
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=self.controlnet,
                vae=self.vae,
                torch_dtype=torch.float16,
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=self.vae,
                torch_dtype=torch.float16,
            )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.enable_model_cpu_offload()
        self.pipe.to("cuda")
        self.proc_elapsed_time = time.time() - self.proc_start_time
        self.proc_start_time = None
        self.models_loaded = True

    def set_control_schema(self, control_schema: str):
        assert control_schema in ["None", "canny", "depth"]
        if self.control_schema is None and (control_schema == "None" or control_schema is None):
            return
        if self.control_schema == control_schema:
            return
        if self.control_schema is not None and control_schema == "None":
            self.control_schema = None
            self.controlnet = None
            self.load_models()
            return
        if self.control_schema != control_schema:
            self.control_schema = control_schema
            self.controlnet = ControlNetModel.from_pretrained(f"diffusers/controlnet-{self.control_schema}-sdxl-1.0", torch_dtype=torch.float16)
            self.load_models()
            return

    def run(
        self,
        control_image: str,
        prompt: str,
        negative_prompt: str = "low quality, bad quality, sketches",
    ):
        self.proc_start_time = time.time()
        image = control_image
        if self.control_schema == "canny":
            image = imgutils.canny_image(control_image)
        if self.control_schema == "depth":
            image = imgutils.get_depth_map(control_image)
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": 50,
            # width=1024,
            # height=1024,
            # guidance_scale=12,
            # target_size=(1024,1024),
            # original_size=(4096,4096)
        }
        if self.controlnet:
            kwargs["image"] = image
            kwargs["controlnet_conditioning_scale"] = self.controlnet_conditioning_scale
        images = self.pipe(**kwargs).images
        images[0].save(f"{prompt}.png")
        self.proc_elapsed_time = time.time() - self.proc_start_time
        self.proc_start_time = None
        return images[0]
