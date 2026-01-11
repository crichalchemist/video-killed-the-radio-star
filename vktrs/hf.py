from pathlib import Path
import torch
from torch import autocast
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

# weird, why didn't this install with vktrs?
#!pip install pytokenizations yt-dlp python-tsp webvtt-py

#use_stability_api = False


# to do: rename "start_schedule" to "strength"
# start_schedule=(1-image_consistency))


class HfHelper:
    def __init__(
        self,
        device = None,
        device_img2img = None,
        device_text2img = None,
        model_path = '.',
        model_id = "CompVis/stable-diffusion-v1-4",
        download=True,
    ):
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not device_img2img:
            device_img2img = device
        if not device_text2img:
            device_text2img = device

        self.device = device
        self.device_img2img = device_img2img
        self.device_text2img = device_text2img
        self.model_path = model_path
        self.model_id = model_id
        self.download = download

        # Lazy loading - pipelines created on first access
        self._text2img = None
        self._img2img = None

        print(f"🎨 Initialized HfHelper on device: {self.device}")

    def _optimize_pipeline(self, pipe):
        """Apply memory optimizations to pipeline"""
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("  ✓ Using xformers (20-30% memory savings)")
        except:
            pipe.enable_attention_slicing()
            print("  ✓ Using attention slicing (fallback)")
        return pipe

    @property
    def img2img(self):
        """Lazy load img2img pipeline on first access"""
        if self._img2img is None:
            print(f"📥 Loading img2img pipeline to {self.device}...")
            if self.download:
                self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=True
                )
                self._img2img = self._img2img.to(self.device)
                self._img2img.save_pretrained(self.model_path)
            else:
                self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model_path,
                    local_files_only=True
                ).to(self.device)

            self._img2img = self._optimize_pipeline(self._img2img)
        return self._img2img

    @property
    def text2img(self):
        """Lazy load text2img pipeline on first access"""
        if self._text2img is None:
            print(f"📥 Loading text2img pipeline to {self.device}...")
            # text2img shares components with img2img
            img2img_pipe = self.img2img  # This will load img2img if not already loaded

            self._text2img = StableDiffusionPipeline(
                vae=img2img_pipe.vae,
                text_encoder=img2img_pipe.text_encoder,
                tokenizer=img2img_pipe.tokenizer,
                unet=img2img_pipe.unet,
                feature_extractor=img2img_pipe.feature_extractor,
                scheduler=img2img_pipe.scheduler,
                safety_checker=img2img_pipe.safety_checker,
            )
            self._text2img = self._optimize_pipeline(self._text2img)
        return self._text2img

    def get_image_for_prompt(
        self,
        prompt,
        **kwargs
    ):
        f = self.text2img if kwargs.get('image') is None else self.img2img
        #if kwargs.get('image_consistency') is not None:
        #kwargs['strength'] = 1- kwargs['image_consistency'] 
        if kwargs.get('start_schedule') is not None:
            #kwargs['strength'] = kwargs['start_schedule']
            kwargs['strength'] = kwargs.pop('start_schedule')
        with autocast(self.device):
            return f(prompt, **kwargs)
