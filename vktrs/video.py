"""
Video compilation and frame processing for VKTRS.

FFmpeg-based video generation with support for:
- Variable frame rates
- Audio synchronization
- Text captions
- Frame upscaling
"""

import subprocess
import textwrap
import glob
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# Frame Processing
# ============================================================================

def add_caption_to_frame(image: Union[np.ndarray, Image.Image],
                        text: str,
                        font_size: int = 20,
                        font_path: Optional[str] = None,
                        position: str = 'bottom',
                        color: tuple = (255, 255, 255),
                        stroke_color: tuple = (0, 0, 0),
                        stroke_width: int = 2,
                        wrap_width: int = 50) -> Image.Image:
    """
    Add text caption to frame.

    Args:
        image: Input image (PIL or numpy)
        text: Caption text
        font_size: Font size in points
        font_path: Path to TTF font (None = default)
        position: 'top', 'bottom', or 'center'
        color: Text color (R, G, B)
        stroke_color: Stroke/outline color
        stroke_width: Stroke width in pixels
        wrap_width: Maximum characters per line

    Returns:
        Image with caption

    Example:
        >>> img = Image.open('frame.png')
        >>> captioned = add_caption_to_frame(
        ...     img,
        ...     "Video killed the radio star",
        ...     font_size=24,
        ...     position='bottom'
        ... )
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    else:
        image = image.copy()

    draw = ImageDraw.Draw(image)

    # Load font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try to use a default system font
            try:
                font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Wrap text
    wrapped_text = textwrap.fill(text, width=wrap_width)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    img_width, img_height = image.size

    x = (img_width - text_width) // 2  # Center horizontally

    if position == 'top':
        y = int(img_height * 0.05)
    elif position == 'center':
        y = (img_height - text_height) // 2
    else:  # bottom
        y = int(img_height * 0.9) - text_height

    # Draw text with stroke
    draw.text(
        (x, y),
        wrapped_text,
        fill=color,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )

    return image


def upscale_frame(image: Union[np.ndarray, Image.Image],
                 scale_factor: int = 2,
                 method: str = 'lanczos') -> Image.Image:
    """
    Upscale frame using high-quality resampling.

    Args:
        image: Input image
        scale_factor: Upscaling factor (2 = 2x size)
        method: Resampling method ('lanczos', 'bicubic', 'bilinear')

    Returns:
        Upscaled image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    new_width = image.width * scale_factor
    new_height = image.height * scale_factor

    # Choose resampling method
    if method == 'lanczos':
        resample = Image.Resampling.LANCZOS
    elif method == 'bicubic':
        resample = Image.Resampling.BICUBIC
    elif method == 'bilinear':
        resample = Image.Resampling.BILINEAR
    else:
        resample = Image.Resampling.LANCZOS

    return image.resize((new_width, new_height), resample)


# ============================================================================
# Video Compilation
# ============================================================================

def compile_video(frame_dir: Path,
                 output_path: Path,
                 audio_path: Optional[Path] = None,
                 fps: int = 12,
                 frame_pattern: str = "frame_%04d.png",
                 crf: int = 18,
                 preset: str = 'medium',
                 upscale: bool = False,
                 upscale_factor: int = 2) -> bool:
    """
    Compile frames into video using FFmpeg.

    Args:
        frame_dir: Directory containing frames
        output_path: Output video path
        audio_path: Audio file path (optional)
        fps: Frames per second
        frame_pattern: Frame filename pattern (printf style)
        crf: Quality (0-51, lower = better, 18 = visually lossless)
        preset: Encoding preset ('ultrafast' to 'veryslow')
        upscale: Whether to upscale frames
        upscale_factor: Upscaling factor if upscale=True

    Returns:
        True if successful

    Example:
        >>> compile_video(
        ...     Path('output/frames'),
        ...     Path('output/video.mp4'),
        ...     audio_path=Path('song.mp3'),
        ...     fps=24,
        ...     crf=18
        ... )
    """
    # Upscale frames if requested
    if upscale:
        print(f"🔍 Upscaling frames {upscale_factor}x...")
        
        # Validate frame directory exists
        if not frame_dir.exists():
            print(f"❌ Frame directory not found: {frame_dir}")
            return False
        
        upscaled_dir = frame_dir.parent / 'frames_upscaled'
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(glob.glob(str(frame_dir / '*.png')))
        
        if not frames:
            print(f"❌ No frames found in {frame_dir}")
            return False

        for i, frame_path in enumerate(frames):
            img = Image.open(frame_path)
            upscaled = upscale_frame(img, upscale_factor, method='lanczos')
            upscaled.save(upscaled_dir / f"frame_{i:04d}.png")

        frame_dir = upscaled_dir
        print(f"  ✓ Upscaled {len(frames)} frames")

    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', str(frame_dir / frame_pattern),
    ]

    # Add audio if provided
    if audio_path:
        cmd.extend(['-i', str(audio_path)])

    cmd.extend([
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',  # Compatibility
    ])

    if audio_path:
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest'  # Match video length to audio
        ])

    cmd.append(str(output_path))

    print(f"🎬 Compiling video: {output_path.name}")
    print(f"   FPS: {fps}, CRF: {crf}, Preset: {preset}")

    # Run FFmpeg with timeout
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timed out after 600 seconds")
        return False

    if result.returncode != 0:
        print("❌ FFmpeg failed:")
        print(result.stderr)
        return False

    print(f"✅ Video compiled: {output_path}")
    return True


def compile_video_with_concat(frame_list: List[Path],
                              durations: List[float],
                              output_path: Path,
                              audio_path: Optional[Path] = None,
                              fps: int = 12,
                              crf: int = 18) -> bool:
    """
    Compile video with variable frame durations using concat demuxer.

    Enables dynamic FPS - each frame can have different duration.

    Args:
        frame_list: List of frame file paths
        durations: Duration for each frame in seconds
        output_path: Output video path
        audio_path: Audio file path (optional)
        fps: Base framerate
        crf: Quality setting

    Returns:
        True if successful

    Example:
        >>> frames = [Path(f'frame{i:03d}.png') for i in range(100)]
        >>> durations = [1/24] * 100  # 24 FPS
        >>> compile_video_with_concat(frames, durations, Path('out.mp4'))
    """
    # Validate inputs
    if len(frame_list) != len(durations):
        print(f"❌ frame_list and durations length mismatch: {len(frame_list)} vs {len(durations)}")
        return False
    
    # Create concat file
    concat_file = output_path.parent / 'concat_list.txt'

    with open(concat_file, 'w') as f:
        for frame_path, duration in zip(frame_list, durations):
            f.write(f"file '{frame_path.absolute()}'\n")
            f.write(f"duration {duration}\n")

        # Last frame needs to be listed again (FFmpeg concat quirk)
        if frame_list:
            f.write(f"file '{frame_list[-1].absolute()}'\n")

    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
    ]

    if audio_path:
        cmd.extend(['-i', str(audio_path)])

    cmd.extend([
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
    ])

    if audio_path:
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest'
        ])

    cmd.append(str(output_path))

    print("🎬 Compiling video with variable frame durations...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timed out after 600 seconds")
        return False
    finally:
        # Always clean up concat file
        concat_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print("❌ FFmpeg failed:")
        print(result.stderr)
        return False

    print(f"✅ Video compiled: {output_path}")
    return True
