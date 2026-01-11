import logging
import threading
import torch
from torch import autocast
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

logger = logging.getLogger(__name__)

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
        
        # Thread locks for lazy initialization
        self._img2img_lock = threading.Lock()
        self._text2img_lock = threading.Lock()

        logger.info(f"🎨 Initialized HfHelper on device: {self.device}")

    def _optimize_pipeline(self, pipe):
        """Apply memory optimizations to pipeline"""
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("  ✓ Using xformers (20-30% memory savings)")
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.info(f"  ✓ Using attention slicing (fallback): {e}")
            pipe.enable_attention_slicing()
        return pipe

    @property
    def img2img(self):
        """Lazy load img2img pipeline on first access"""
        if self._img2img is None:
            with self._img2img_lock:
                # Double-check pattern
                if self._img2img is None:
                    logger.info(f"📥 Loading img2img pipeline to {self.device_img2img}...")
                    if self.download:
                        self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                            self.model_id,
                            revision="fp16",
                            torch_dtype=torch.float16,
                            use_auth_token=True
                        )
                        self._img2img = self._img2img.to(self.device_img2img)
                        self._img2img.save_pretrained(self.model_path)
                    else:
                        self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                            self.model_path,
                            local_files_only=True
                        ).to(self.device_img2img)

                    self._img2img = self._optimize_pipeline(self._img2img)
        return self._img2img

    @property
    def text2img(self):
        """Lazy load text2img pipeline on first access"""
        if self._text2img is None:
            with self._text2img_lock:
                # Double-check pattern
                if self._text2img is None:
                    logger.info(f"📥 Loading text2img pipeline to {self.device_text2img}...")
                    
                    # Create text2img from pretrained rather than sharing components
                    # This allows proper device placement
                    if self.download:
                        self._text2img = StableDiffusionPipeline.from_pretrained(
                            self.model_id,
                            revision="fp16",
                            torch_dtype=torch.float16,
                            use_auth_token=True
                        )
                        self._text2img = self._text2img.to(self.device_text2img)
                    else:
                        self._text2img = StableDiffusionPipeline.from_pretrained(
                            self.model_path,
                            local_files_only=True
                        ).to(self.device_text2img)
                    
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
