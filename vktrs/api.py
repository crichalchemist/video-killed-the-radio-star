import io
import os
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
import warnings

import torch
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation


@lru_cache(maxsize=1)
def get_api_client():
    """
    Get or create singleton StabilityInference client.
    Uses LRU cache for thread-safe singleton pattern.

    Returns:
        StabilityInference: Reused API client

    Raises:
        ValueError: If STABILITY_KEY environment variable not set
    """
    api_key = os.environ.get('STABILITY_KEY')
    if not api_key:
        raise ValueError(
            "STABILITY_KEY environment variable not set. "
            "Get your API key at https://platform.stability.ai/account/keys"
        )

    print("🔗 Creating Stability API client (reused for all requests)")
    return client.StabilityInference(
        key=api_key,
        verbose=False,
        engine="stable-diffusion-xl-1024-v1-0"
    )


def get_image_for_prompt(prompt, max_retries=3, **kargs):
    stability_api = get_api_client()  # Reuses existing connection


    # auto-retry if mitigation triggered
    while max_retries:

        try:
            answers = stability_api.generate(prompt=prompt, **kargs)

            response = process_response(answers)

            for img in response:

                yield img

            break # whoops... this breaks us out of the while loop, not the for loop.
        except RuntimeError:
            print("runtime error")
            max_retries -= 1
            warnings.warn(f"mitigation triggered, retries remaining: {max_retries}")


def process_response(answers):

    # iterating over the generator produces the api response
    for resp in answers:

        for artifact in resp.artifacts:

            #print(artifact.finish_reason)
            if artifact.finish_reason == generation.FILTER:

                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
                #raise RuntimeError
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                yield img


def validate_environment():
    """
    Validate required environment variables and dependencies.
    Returns dict with capability flags.
    """
    capabilities = {
        'stability_api': 'STABILITY_KEY' in os.environ,
        'cuda_available': torch.cuda.is_available(),
        'xformers_available': False,
        'torch_version': torch.__version__,
    }

    try:
        import xformers
        capabilities['xformers_available'] = True
        capabilities['xformers_version'] = xformers.__version__
    except ImportError:
        pass

    if capabilities['cuda_available']:
        capabilities['gpu_name'] = torch.cuda.get_device_name(0)
        capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return capabilities


def print_system_info():
    """Print system capabilities for debugging"""
    caps = validate_environment()

    print("=" * 60)
    print("🎬 VKTRS System Capabilities")
    print("=" * 60)
    print(f"🔧 PyTorch version: {caps['torch_version']}")
    print(f"🎮 CUDA available: {caps['cuda_available']}")

    if caps['cuda_available']:
        print(f"   GPU: {caps['gpu_name']}")
        print(f"   Memory: {caps['gpu_memory_gb']:.1f} GB")
    else:
        print("   Using CPU (slower, but works!)")

    print(f"⚡ xformers: {caps['xformers_available']}")
    if caps['xformers_available']:
        print(f"   Version: {caps.get('xformers_version', 'unknown')}")
        print("   ✓ Memory-efficient attention enabled (20-30% savings)")
    else:
        print("   ℹ️  Install xformers for 20-30% memory savings:")
        print("   pip install xformers")

    print(f"🔑 Stability API: {caps['stability_api']}")
    if caps['stability_api']:
        print("   ✓ API key found")
    else:
        print("   ⚠️  STABILITY_KEY not set (required for DreamStudio API)")
        print("   Get your key at: https://platform.stability.ai/account/keys")

    print("=" * 60)
    print()

    # Provide recommendations
    if not caps['cuda_available'] and not caps['stability_api']:
        print("⚠️  WARNING: No GPU and no Stability API key detected!")
        print("   You have two options:")
        print("   1. Set STABILITY_KEY to use DreamStudio API (recommended for CPU)")
        print("   2. Use GPU runtime in Google Colab for local generation")
        print()
    elif not caps['cuda_available']:
        print("ℹ️  INFO: CPU mode detected")
        print("   Using Stability API for image generation (recommended)")
        print("   Local CPU generation will be very slow")
        print()

    return caps
