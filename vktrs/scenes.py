"""
Scene creation and management for VKTRS.

Scenes are the fundamental timeline units in VKTRS, typically derived from
lyrical segments or musical structure.
"""

from typing import List, Dict
import copy
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def create_scenes_from_lyrics(transcription: Dict,
                              min_scene_duration: float = 2.0,
                              max_scene_duration: float = 10.0) -> List[Dict]:
    """
    Create scenes from Whisper transcription segments.

    Args:
        transcription: Whisper output with 'segments'
        min_scene_duration: Minimum scene length in seconds
        max_scene_duration: Maximum scene length in seconds

    Returns:
        List of scene dicts with 'start_time', 'end_time', 'text'
    """
    # Validate input
    if not isinstance(transcription, dict):
        raise TypeError(f"transcription must be a dict, got {type(transcription).__name__}")
    if 'segments' not in transcription:
        raise ValueError("transcription must contain 'segments' key")
    if min_scene_duration <= 0:
        raise ValueError(f"min_scene_duration must be positive, got {min_scene_duration}")
    if max_scene_duration <= 0:
        raise ValueError(f"max_scene_duration must be positive, got {max_scene_duration}")
    if min_scene_duration > max_scene_duration:
        raise ValueError(f"min_scene_duration ({min_scene_duration}) must be <= max_scene_duration ({max_scene_duration})")
    if not transcription['segments']:
        return []
    
    scenes = []

    for segment in transcription['segments']:
        # Validate segment has required keys
        if 'start' not in segment or 'end' not in segment:
            raise ValueError(f"Segment missing required 'start' or 'end' key: {segment}")
        if 'text' not in segment:
            # Use empty text if missing
            segment['text'] = ''
        
        scene = {
            'start_time': segment['start'],
            'end_time': segment['end'],
            'text': segment['text'].strip(),
            'prompt': None  # To be filled later
        }
        scenes.append(scene)

    # Merge short scenes
    scenes = merge_short_scenes(scenes, min_duration=min_scene_duration)

    # Subdivide long scenes
    scenes = subdivide_long_scenes(scenes, max_duration=max_scene_duration)

    print(f"📝 Created {len(scenes)} scenes from {len(transcription['segments'])} lyric segments")
    return scenes


def subdivide_long_scenes(scenes: List[Dict],
                          max_duration: float = 10.0) -> List[Dict]:
    """
    Split scenes longer than max_duration into equal parts.

    Args:
        scenes: List of scene dicts
        max_duration: Maximum scene duration in seconds

    Returns:
        List of scenes with long scenes subdivided
    """
    if not isinstance(scenes, list):
        raise TypeError(f"scenes must be a list, got {type(scenes).__name__}")
    if max_duration <= 0:
        raise ValueError(f"max_duration must be positive, got {max_duration}")
    
    result = []

    for scene in scenes:
        # Validate scene has required keys
        if 'start_time' not in scene or 'end_time' not in scene:
            raise ValueError(f"Scene missing required 'start_time' or 'end_time' key: {scene}")
        
        duration = scene['end_time'] - scene['start_time']

        if duration <= max_duration:
            result.append(scene)
        else:
            # Split into equal parts
            n_parts = int(np.ceil(duration / max_duration))
            part_duration = duration / n_parts

            for i in range(n_parts):
                part = copy.deepcopy(scene)
                part['start_time'] = scene['start_time'] + i * part_duration
                part['end_time'] = scene['start_time'] + (i + 1) * part_duration

                # Annotate subdivided scenes
                if 'text' in scene:
                    part['text'] = f"{scene['text']} (part {i+1}/{n_parts})"

                result.append(part)

    return result


def merge_short_scenes(scenes: List[Dict],
                       min_duration: float = 2.0) -> List[Dict]:
    """
    Merge scenes shorter than min_duration with neighbors.

    Args:
        scenes: List of scene dicts
        min_duration: Minimum scene duration in seconds

    Returns:
        List of scenes with short scenes merged
    """
    if not isinstance(scenes, list):
        raise TypeError(f"scenes must be a list, got {type(scenes).__name__}")
    if min_duration <= 0:
        raise ValueError(f"min_duration must be positive, got {min_duration}")
    if not scenes:
        return scenes

    result = [scenes[0]]

    for scene in scenes[1:]:
        last_scene = result[-1]
        
        # Validate scenes have required keys
        if 'start_time' not in last_scene or 'end_time' not in last_scene:
            raise ValueError(f"Scene missing required 'start_time' or 'end_time' key: {last_scene}")
        if 'start_time' not in scene or 'end_time' not in scene:
            raise ValueError(f"Scene missing required 'start_time' or 'end_time' key: {scene}")
        
        last_duration = last_scene['end_time'] - last_scene['start_time']
        curr_duration = scene['end_time'] - scene['start_time']

        if last_duration < min_duration or curr_duration < min_duration:
            # Merge with previous
            result[-1] = {
                'start_time': last_scene['start_time'],
                'end_time': scene['end_time'],
                'text': f"{last_scene.get('text', '')} {scene.get('text', '')}".strip()
            }
        else:
            result.append(scene)

    return result


def assign_themes(scenes: List[Dict],
                 audio_analysis: Dict,
                 n_themes: int = 5) -> Dict[str, List[int]]:
    """
    Group scenes into themes based on musical similarity.

    Uses hierarchical clustering on CQT features to identify
    musically similar sections.

    Args:
        scenes: List of scene dicts
        audio_analysis: Output from analyze_audio_structure()
        n_themes: Number of theme clusters

    Returns:
        Dict mapping theme IDs to scene indices:
            {'theme_0': [0, 3, 5], 'theme_1': [1, 2, 4], ...}
    """
    # Validate inputs
    if not isinstance(scenes, list):
        raise TypeError(f"scenes must be a list, got {type(scenes).__name__}")
    if not scenes:
        raise ValueError("scenes must not be empty")
    if not isinstance(audio_analysis, dict):
        raise TypeError(f"audio_analysis must be a dict, got {type(audio_analysis).__name__}")
    if n_themes <= 0:
        raise ValueError(f"n_themes must be positive, got {n_themes}")
    if 'cqt' not in audio_analysis:
        raise ValueError("audio_analysis must contain 'cqt' key")
    if 'sr' not in audio_analysis:
        raise ValueError("audio_analysis must contain 'sr' key")
    
    cqt = audio_analysis['cqt']
    sr = audio_analysis['sr']
    hop_length = audio_analysis.get('hop_length', 512)

    # Extract features for each scene
    scene_features = []
    for scene in scenes:
        # Validate scene structure
        if 'start_time' not in scene or 'end_time' not in scene:
            raise ValueError(f"Scene missing required 'start_time' or 'end_time' key: {scene}")
        # Validate scene values
        if not isinstance(scene['start_time'], (int, float)) or not isinstance(scene['end_time'], (int, float)):
            raise ValueError(f"Scene times must be numeric: {scene}")
        if scene['start_time'] >= scene['end_time']:
            raise ValueError(f"Scene start_time ({scene['start_time']}) must be < end_time ({scene['end_time']})")
        
        # Get CQT slice for this scene
        start_frame = int(scene['start_time'] * sr / hop_length)
        end_frame = int(scene['end_time'] * sr / hop_length)

        # Clip to CQT bounds
        end_frame = min(end_frame, cqt.shape[1])

        if start_frame >= cqt.shape[1]:
            # Scene beyond audio (shouldn't happen with validation)
            scene_cqt = cqt[:, -1:]
        else:
            scene_cqt = cqt[:, start_frame:end_frame]

        # Summarize as mean + std
        feature = np.concatenate([
            np.mean(scene_cqt, axis=1),
            np.std(scene_cqt, axis=1)
        ])

        scene_features.append(feature)

    # Cluster scenes
    X = np.array(scene_features)
    n_clusters = min(n_themes, len(scenes))

    if n_clusters < 2:
        # Not enough scenes to cluster
        return {'theme_0': list(range(len(scenes)))}

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    # Group by theme
    themes = {}
    for i, label in enumerate(labels):
        theme_id = f"theme_{label}"
        if theme_id not in themes:
            themes[theme_id] = []
        themes[theme_id].append(i)

    print(f"🎨 Assigned {len(scenes)} scenes to {len(themes)} themes")
    for theme_id, scene_ids in themes.items():
        print(f"  {theme_id}: {len(scene_ids)} scenes")

    return themes


def create_scenes_from_structure(audio_analysis: Dict) -> List[Dict]:
    """
    Create scenes directly from musical structure analysis.

    Useful when lyrics are unavailable or for instrumental tracks.

    Args:
        audio_analysis: Output from analyze_audio_structure()

    Returns:
        List of scene dicts based on structural segments
    """
    # Validate audio_analysis
    if not isinstance(audio_analysis, dict):
        raise TypeError(f"audio_analysis must be a dict, got {type(audio_analysis).__name__}")
    if 'segments' not in audio_analysis:
        raise ValueError("audio_analysis must contain 'segments' key")
    if 'sr' not in audio_analysis:
        raise ValueError("audio_analysis must contain 'sr' key")
    
    segments = audio_analysis['segments']
    sr = audio_analysis['sr']
    hop_length = audio_analysis.get('hop_length', 512)

    scenes = []
    for i, segment in enumerate(segments):
        # Validate segment has required keys
        if 'start_frame' not in segment or 'end_frame' not in segment or 'label' not in segment:
            raise ValueError(f"Segment {i} missing required keys ('start_frame', 'end_frame', 'label'): {segment}")
        
        # Convert frames to time
        start_time = segment['start_frame'] * hop_length / sr
        end_time = segment['end_frame'] * hop_length / sr

        scene = {
            'start_time': float(start_time),
            'end_time': float(end_time),
            'text': f"Segment {i} (label {segment['label']})",
            'structural_label': segment['label']
        }
        scenes.append(scene)

    print(f"📝 Created {len(scenes)} scenes from musical structure")
    return scenes
