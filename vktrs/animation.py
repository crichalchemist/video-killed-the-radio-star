"""
Animation parameter calculation and keyframe interpolation for VKTRS.

Handles camera motion (zoom, pan, rotation), parameter interpolation,
and easing functions.
"""

import numpy as np
from typing import List, Dict, Callable, Optional, Union


# ============================================================================
# Interpolation & Easing
# ============================================================================

def linear_interpolate(t: float, start: float, end: float) -> float:
    """Linear interpolation between start and end."""
    return start + t * (end - start)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out easing function."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out."""
    if t < 0.5:
        return 2 * t * t
    else:
        return 1 - pow(-2 * t + 2, 2) / 2


EASING_FUNCTIONS = {
    'linear': lambda t: t,
    'ease_in_quad': ease_in_quad,
    'ease_out_quad': ease_out_quad,
    'ease_in_out_quad': ease_in_out_quad,
    'ease_in_out_cubic': ease_in_out_cubic,
}


def interpolate_keyframes(keyframes: List[Dict],
                         num_frames: int,
                         easing: str = 'linear') -> np.ndarray:
    """
    Interpolate between keyframes.

    Args:
        keyframes: List of {frame: int, value: float} dicts
        num_frames: Total number of frames
        easing: Easing function name

    Returns:
        Array of interpolated values for each frame

    Example:
        >>> keyframes = [
        ...     {'frame': 0, 'value': 1.0},
        ...     {'frame': 50, 'value': 1.5},
        ...     {'frame': 100, 'value': 1.0}
        ... ]
        >>> values = interpolate_keyframes(keyframes, 101, 'ease_in_out_quad')
    """
    if not keyframes:
        return np.zeros(num_frames)

    # Sort by frame
    keyframes = sorted(keyframes, key=lambda k: k['frame'])

    # Get easing function
    easing_func = EASING_FUNCTIONS.get(easing, lambda t: t)

    # Interpolate
    values = np.zeros(num_frames)

    for i in range(num_frames):
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf in keyframes:
            if kf['frame'] <= i:
                prev_kf = kf
            if kf['frame'] >= i and next_kf is None:
                next_kf = kf
                break

        if prev_kf is None:
            # Before first keyframe
            values[i] = keyframes[0]['value']
        elif next_kf is None:
            # After last keyframe
            values[i] = keyframes[-1]['value']
        elif prev_kf['frame'] == next_kf['frame']:
            # On keyframe
            values[i] = prev_kf['value']
        else:
            # Between keyframes
            frame_range = next_kf['frame'] - prev_kf['frame']
            t = (i - prev_kf['frame']) / frame_range
            t_eased = easing_func(t)
            values[i] = linear_interpolate(
                t_eased,
                prev_kf['value'],
                next_kf['value']
            )

    return values


# ============================================================================
# Camera Motion
# ============================================================================

def calculate_zoom(start_zoom: float,
                  end_zoom: float,
                  num_frames: int,
                  easing: str = 'ease_in_out_quad') -> np.ndarray:
    """
    Calculate smooth zoom curve.

    Args:
        start_zoom: Initial zoom (1.0 = no zoom)
        end_zoom: Final zoom
        num_frames: Number of frames
        easing: Easing function

    Returns:
        Zoom values for each frame
    """
    keyframes = [
        {'frame': 0, 'value': start_zoom},
        {'frame': num_frames - 1, 'value': end_zoom}
    ]
    return interpolate_keyframes(keyframes, num_frames, easing)


def calculate_pan(start_x: float,
                 start_y: float,
                 end_x: float,
                 end_y: float,
                 num_frames: int,
                 easing: str = 'ease_in_out_quad') -> tuple:
    """
    Calculate smooth 2D pan curve.

    Args:
        start_x: Initial X position
        start_y: Initial Y position
        end_x: Final X position
        end_y: Final Y position
        num_frames: Number of frames
        easing: Easing function

    Returns:
        (x_values, y_values) arrays
    """
    x_keyframes = [
        {'frame': 0, 'value': start_x},
        {'frame': num_frames - 1, 'value': end_x}
    ]
    y_keyframes = [
        {'frame': 0, 'value': start_y},
        {'frame': num_frames - 1, 'value': end_y}
    ]

    x_values = interpolate_keyframes(x_keyframes, num_frames, easing)
    y_values = interpolate_keyframes(y_keyframes, num_frames, easing)

    return x_values, y_values


def calculate_rotation(start_angle: float,
                      end_angle: float,
                      num_frames: int,
                      easing: str = 'ease_in_out_quad') -> np.ndarray:
    """
    Calculate smooth rotation curve.

    Args:
        start_angle: Initial angle in degrees
        end_angle: Final angle in degrees
        num_frames: Number of frames
        easing: Easing function

    Returns:
        Rotation values for each frame
    """
    keyframes = [
        {'frame': 0, 'value': start_angle},
        {'frame': num_frames - 1, 'value': end_angle}
    ]
    return interpolate_keyframes(keyframes, num_frames, easing)


def calculate_camera_motion(scene_duration: float,
                           fps: int,
                           zoom: Optional[Dict] = None,
                           pan: Optional[Dict] = None,
                           rotation: Optional[Dict] = None) -> Dict:
    """
    Calculate complete camera motion for a scene.

    Args:
        scene_duration: Scene duration in seconds
        fps: Frames per second
        zoom: {'start': float, 'end': float, 'easing': str}
        pan: {'start_x': float, 'start_y': float, 'end_x': float, 'end_y': float, 'easing': str}
        rotation: {'start': float, 'end': float, 'easing': str}

    Returns:
        Dict with 'zoom', 'pan_x', 'pan_y', 'rotation' arrays

    Example:
        >>> motion = calculate_camera_motion(
        ...     scene_duration=5.0,
        ...     fps=24,
        ...     zoom={'start': 1.0, 'end': 1.2, 'easing': 'ease_in_out_quad'},
        ...     pan={'start_x': 0, 'start_y': 0, 'end_x': 10, 'end_y': -5, 'easing': 'linear'}
        ... )
    """
    num_frames = int(scene_duration * fps)

    result = {}

    # Zoom
    if zoom:
        result['zoom'] = calculate_zoom(
            zoom.get('start', 1.0),
            zoom.get('end', 1.0),
            num_frames,
            zoom.get('easing', 'ease_in_out_quad')
        )
    else:
        result['zoom'] = np.ones(num_frames)

    # Pan
    if pan:
        result['pan_x'], result['pan_y'] = calculate_pan(
            pan.get('start_x', 0),
            pan.get('start_y', 0),
            pan.get('end_x', 0),
            pan.get('end_y', 0),
            num_frames,
            pan.get('easing', 'ease_in_out_quad')
        )
    else:
        result['pan_x'] = np.zeros(num_frames)
        result['pan_y'] = np.zeros(num_frames)

    # Rotation
    if rotation:
        result['rotation'] = calculate_rotation(
            rotation.get('start', 0),
            rotation.get('end', 0),
            num_frames,
            rotation.get('easing', 'ease_in_out_quad')
        )
    else:
        result['rotation'] = np.zeros(num_frames)

    return result


# ============================================================================
# Parameter Resolution
# ============================================================================

def resolve_parameter_expression(expression: str,
                                frame_id: int,
                                scene_id: int,
                                total_frames: int,
                                **context) -> float:
    """
    Evaluate mathematical expression for animation parameter.

    Supports temporal variables and custom context.

    Args:
        expression: Math expression (e.g., "1.0 + 0.2 * sin(t * 2 * pi)")
        frame_id: Current frame index
        scene_id: Current scene index
        total_frames: Total frames in scene
        **context: Additional variables (signals, etc.)

    Returns:
        Evaluated parameter value

    Example:
        >>> # Oscillating zoom
        >>> zoom = resolve_parameter_expression(
        ...     "1.0 + 0.1 * sin(t * 4 * pi)",
        ...     frame_id=50,
        ...     scene_id=0,
        ...     total_frames=100
        ... )
    """
    import math

    # Normalized time (0 to 1 within scene)
    t = frame_id / max(total_frames - 1, 1)

    # Build evaluation context
    eval_context = {
        't': t,
        'frame': frame_id,
        'scene': scene_id,
        'total_frames': total_frames,
        'pi': math.pi,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'abs': abs,
        'min': min,
        'max': max,
        'pow': pow,
        **context
    }

    try:
        result = eval(expression, {"__builtins__": {}}, eval_context)
        return float(result)
    except Exception as e:
        print(f"⚠️  Error evaluating '{expression}': {e}")
        return 0.0
