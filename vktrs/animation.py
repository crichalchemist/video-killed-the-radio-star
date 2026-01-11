"""
Keyframe interpolation and camera motion for VKTRS.

Optional dependencies for expression evaluation:
- simpleeval: Safe expression evaluation (recommended)
- asteval: Fallback expression evaluator

Install with: pip install simpleeval asteval

Functions requiring expression evaluation:
- resolve_parameter_expression(): Evaluates dynamic parameter expressions
"""

import logging
import numbers
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeAlias

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


logger = logging.getLogger(__name__)

_NUMPY_UNAVAILABLE = "NumPy not available"


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


EASING_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    'linear': lambda t: t,
    'ease_in_quad': ease_in_quad,
    'ease_out_quad': ease_out_quad,
    'ease_in_out_quad': ease_in_out_quad,
    'ease_in_out_cubic': ease_in_out_cubic,
}


Keyframe: TypeAlias = Dict[str, Any]


def _find_surrounding_keyframes(
    keyframes: Sequence[dict[str, Any]],
    frame_idx: int,
) -> Tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Find keyframes surrounding a given frame index."""
    prev_kf: Optional[dict[str, Any]] = None
    next_kf: Optional[dict[str, Any]] = None
    
    for kf in keyframes:
        if kf['frame'] <= frame_idx:
            prev_kf = kf
        elif next_kf is None:
            next_kf = kf
            break
    
    return prev_kf, next_kf


def _interpolate_frame(
    frame_idx: int,
    keyframes: Sequence[dict[str, Any]],
    easing_func: Callable[[float], float],
) -> float:
    """Interpolate value for a single frame."""
    prev_kf, next_kf = _find_surrounding_keyframes(keyframes, frame_idx)
    
    if prev_kf is None:
        return float(keyframes[0]['value'])
    if next_kf is None:
        return float(keyframes[-1]['value'])
    if prev_kf['frame'] == next_kf['frame']:
        return float(prev_kf['value'])
    
    frame_range = float(next_kf['frame'] - prev_kf['frame'])
    t = float(frame_idx - prev_kf['frame']) / frame_range
    t_eased = easing_func(t)
    return linear_interpolate(t_eased, float(prev_kf['value']), float(next_kf['value']))


def interpolate_keyframes(keyframes: Sequence[dict[str, Any]],
                         num_frames: int,
                         easing: str = 'linear') -> Any:
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
    # Validate inputs
    if num_frames <= 0:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames}")

    if not keyframes:
        if np is None:
            raise RuntimeError(_NUMPY_UNAVAILABLE)
        return np.zeros(num_frames)

    # Sort by frame
    keyframes = sorted(keyframes, key=lambda k: k['frame'])

    # Get easing function
    easing_func = EASING_FUNCTIONS.get(easing, lambda t: t)

    # Interpolate
    if np is None:
        raise RuntimeError(_NUMPY_UNAVAILABLE)
    values = np.zeros(num_frames, dtype=float)

    for i in range(num_frames):
        values[i] = _interpolate_frame(i, keyframes, easing_func)

    return values


# ============================================================================
# Camera Motion
# ============================================================================

def calculate_zoom(start_zoom: float,
                  end_zoom: float,
                  num_frames: int,
                  easing: str = 'ease_in_out_quad') -> Any:
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
                 easing: str = 'ease_in_out_quad') -> Tuple[Any, Any]:
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
                      easing: str = 'ease_in_out_quad') -> Any:
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


def calculate_camera_motion(
    scene_duration: float,
    fps: int,
    zoom: Optional[Dict[str, Any]] = None,
    pan: Optional[Dict[str, Any]] = None,
    rotation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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
    if scene_duration <= 0:
        raise ValueError(f"scene_duration must be positive, got {scene_duration}")
    if not isinstance(fps, numbers.Integral):
        raise TypeError(f"fps must be an integer-like value, got {type(fps).__name__}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    num_frames = _compute_num_frames(scene_duration, fps)
    return _build_camera_result(num_frames, zoom, pan, rotation)


def _compute_num_frames(scene_duration: float, fps: int) -> int:
    """Compute number of frames from duration and fps."""
    if np is None:
        return int(scene_duration * fps)
    return max(1, int(np.ceil(scene_duration * fps)))


def _build_camera_result(
    num_frames: int,
    zoom: Optional[Dict[str, Any]],
    pan: Optional[Dict[str, Any]],
    rotation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build camera motion result dict."""
    result: Dict[str, Any] = {}
    
    _add_zoom_to_result(result, num_frames, zoom)
    _add_pan_to_result(result, num_frames, pan)
    _add_rotation_to_result(result, num_frames, rotation)
    
    return result


def _add_zoom_to_result(
    result: Dict[str, Any],
    num_frames: int,
    zoom: Optional[Dict[str, Any]],
) -> None:
    """Add zoom to camera result."""
    if zoom:
        zoom_easing = str(zoom.get('easing', 'ease_in_out_quad'))
        result['zoom'] = calculate_zoom(
            zoom.get('start', 1.0),
            zoom.get('end', 1.0),
            num_frames,
            zoom_easing
        )
    else:
        if np is None:
            raise RuntimeError(_NUMPY_UNAVAILABLE)
        result['zoom'] = np.ones(num_frames)


def _add_pan_to_result(
    result: Dict[str, Any],
    num_frames: int,
    pan: Optional[Dict[str, Any]],
) -> None:
    """Add pan to camera result."""
    if pan:
        pan_easing = str(pan.get('easing', 'ease_in_out_quad'))
        result['pan_x'], result['pan_y'] = calculate_pan(
            pan.get('start_x', 0),
            pan.get('start_y', 0),
            pan.get('end_x', 0),
            pan.get('end_y', 0),
            num_frames,
            pan_easing
        )
    else:
        if np is None:
            raise RuntimeError(_NUMPY_UNAVAILABLE)
        result['pan_x'] = np.zeros(num_frames)
        result['pan_y'] = np.zeros(num_frames)


def _add_rotation_to_result(
    result: Dict[str, Any],
    num_frames: int,
    rotation: Optional[Dict[str, Any]],
) -> None:
    """Add rotation to camera result."""
    if rotation:
        rotation_easing = str(rotation.get('easing', 'ease_in_out_quad'))
        result['rotation'] = calculate_rotation(
            rotation.get('start', 0),
            rotation.get('end', 0),
            num_frames,
            rotation_easing
        )
    else:
        if np is None:
            raise RuntimeError(_NUMPY_UNAVAILABLE)
        result['rotation'] = np.zeros(num_frames)


# ============================================================================
# Parameter Resolution
# ============================================================================

def resolve_parameter_expression(
    expression: str,
    frame_id: Optional[int] = None,
    scene_id: Optional[int] = None,
    total_frames: int = 1,
    *,
    t: Optional[float] = None,
    frame: Optional[int] = None,
    scene: Optional[int] = None,
    **context: Any,
) -> float:
    """
    Evaluate dynamic parameter expressions with math and context variables.

    Supports expressions like:
    - "sin(t * 2 * pi)"
    - "frame / total_frames"
    - "scene * 0.5 + 1"

    Args:
        expression: Math expression string
        frame_id: Legacy frame index (used if ``frame`` not provided)
        scene_id: Legacy scene index (used if ``scene`` not provided)
        total_frames: Total frame count (used to derive ``t`` when not provided)
        t: Normalized time in [0, 1]; auto-computed from frame/total_frames if omitted
        frame: Current frame number (preferred over frame_id)
        scene: Current scene index (preferred over scene_id)
        **context: Additional variables for expression

    Returns:
        Evaluated float value. Returns 0.0 on expected evaluation errors.

    Behavior:
        - simpleeval errors trigger fallback to asteval
        - asteval errors are logged and return 0.0
        - ImportError if neither dependency available
        - Unexpected exceptions propagate to surface bugs
    """
    import math

    # Resolve frame/scene values with backward compatibility
    frame_val = frame if frame is not None else (frame_id or 0)
    scene_val = scene if scene is not None else (scene_id or 0)

    # Derive normalized time if not explicitly provided
    if t is None:
        t = frame_val / max(total_frames - 1, 1) if total_frames > 1 else 0.0

    # Build evaluation context (include legacy names for compatibility)
    eval_context: Dict[str, Any] = {
        't': t,
        'frame': frame_val,
        'frame_id': frame_val,
        'scene': scene_val,
        'scene_id': scene_val,
        'total_frames': total_frames,
        'pi': math.pi,
        'e': math.e,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'abs': abs,
        'min': min,
        'max': max,
        **context,
    }

    return _evaluate_with_fallback(expression, eval_context)


def _evaluate_with_fallback(expression: str, eval_context: Dict[str, Any]) -> float:
    """
    Attempt expression evaluation with simpleeval, fall back to asteval.
    
    Returns: float result, or 0.0 on error
    """
    # Try simpleeval first (safer, faster)
    try:
        from simpleeval import simple_eval
        result = simple_eval(expression, names=eval_context)
        return float(result)
    except ImportError:
        logger.debug("simpleeval not installed; falling back to asteval")
    except Exception as exc:
        logger.debug(
            "simpleeval failed for expression '%s' with error: %s; falling back to asteval",
            expression,
            exc,
        )

    # Fallback to asteval
    try:
        from asteval import Interpreter
        aeval = Interpreter()
        aeval.symtable.update(eval_context)
        result = aeval(expression)

        if aeval.error:
            def _err_to_str(err: Any) -> str:
                return str(err.get_error()) if hasattr(err, "get_error") else str(err)
            formatted_errors = ", ".join(_err_to_str(err) for err in aeval.error)
            logger.warning(
                "asteval reported errors for expression '%s': %s",
                expression,
                formatted_errors,
            )
            return 0.0
        return float(result)
    except ImportError:
        raise ImportError(
            "Expression evaluation requires simpleeval or asteval. "
            "Install with: pip install simpleeval asteval"
        )
    except (ValueError, TypeError, NameError, ArithmeticError, SyntaxError):
        logger.exception("Error evaluating expression '%s'", expression)
        return 0.0
