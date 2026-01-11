"""
Scene creation and management for VKTRS.

Scenes are the fundamental timeline units in VKTRS, typically derived from
lyrical segments or musical structure.
"""

from typing import List, Dict, Optional
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
    scenes = []

    for segment in transcription['segments']:
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
    result = []

    for scene in scenes:
        duration = scene['end_time'] - scene['start_time']

        if duration <= max_duration:
            result.append(scene)
        else:
            # Split into equal parts
            n_parts = int(np.ceil(duration / max_duration))
            part_duration = duration / n_parts

            for i in range(n_parts):
                part = scene.copy()
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
    if not scenes:
        return scenes

    result = [scenes[0]]

    for scene in scenes[1:]:
        last_scene = result[-1]
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
    cqt = audio_analysis['cqt']
    sr = audio_analysis['sr']
    hop_length = audio_analysis.get('hop_length', 512)

    # Extract features for each scene
    scene_features = []
    for scene in scenes:
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
    segments = audio_analysis['segments']
    sr = audio_analysis['sr']
    hop_length = audio_analysis.get('hop_length', 512)

    scenes = []
    for i, segment in enumerate(segments):
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
