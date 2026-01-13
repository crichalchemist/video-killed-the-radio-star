import io
import os
import logging
from functools import lru_cache
from PIL import Image
import warnings

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

logger = logging.getLogger(__name__)


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

    logger.info("🔗 Creating Stability API client (reused for all requests)")
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
        'has_stability_key': 'STABILITY_KEY' in os.environ,
        'cuda_available': False,
        'xformers_available': False,
    }
    
    # Try to import torch
    try:
        import torch
        capabilities['torch_version'] = torch.__version__
        capabilities['cuda_available'] = torch.cuda.is_available()
        
        if capabilities['cuda_available']:
            try:
                capabilities['gpu_name'] = torch.cuda.get_device_name(0)
                capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                logger.exception("Error retrieving CUDA device information")
                capabilities['gpu_name'] = None
                capabilities['gpu_memory_gb'] = None
    except ImportError:
        logger.info("torch not available, skipping torch-specific checks")
        capabilities['torch_version'] = None

    try:
        import xformers
        capabilities['xformers_available'] = True
        capabilities['xformers_version'] = xformers.__version__
    except ImportError:
        pass

    return capabilities


def print_and_get_system_info():
    """Print system capabilities for debugging and return the capabilities dict"""
    caps = validate_environment()

    print("=" * 60)
    print("🎬 VKTRS System Capabilities")
    print("=" * 60)
    if caps.get('torch_version'):
        print(f"🔧 PyTorch version: {caps['torch_version']}")
    print(f"🎮 CUDA available: {caps['cuda_available']}")

    if caps['cuda_available']:
        gpu_name = caps.get('gpu_name')
        gpu_memory = caps.get('gpu_memory_gb')
        if gpu_name is not None:
            print(f"   GPU: {gpu_name}")
        if gpu_memory is not None:
            print(f"   Memory: {gpu_memory:.1f} GB")
        if gpu_name is None or gpu_memory is None:
            print("   (Unable to retrieve GPU details)")
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


# Backward compatibility alias
print_system_info = print_and_get_system_info
