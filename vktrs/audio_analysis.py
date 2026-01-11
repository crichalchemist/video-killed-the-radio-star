"""
Audio analysis and structure detection for VKTRS.

This module provides functions for:
- Musical structure analysis (tempo, key, segments)
- Stem separation (vocals, drums, bass, other)
- Audio feature extraction
- Spectral analysis

Based on McFee and Ellis 2014 research for structure detection.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
from sklearn.cluster import SpectralClustering


# ============================================================================
# Audio Structure Analysis
# ============================================================================

def analyze_audio_structure(audio_fpath: str,
                           sr: int = 22050,
                           n_fft: int = 2048,
                           hop_length: int = 512,
                           n_segments: Optional[int] = None) -> Dict:
    """
    Analyze musical structure using constant-Q transform and spectral clustering.

    Based on McFee and Ellis 2014 paper on structural segmentation.

    Args:
        audio_fpath: Path to audio file
        sr: Sample rate for analysis
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_segments: Number of segments (auto-detected if None)

    Returns:
        dict with keys:
            - 'tempo': Estimated tempo in BPM
            - 'key': Estimated musical key
            - 'segments': List of structural segments
            - 'beats': Beat frame indices
            - 'cqt': Constant-Q transform matrix
            - 'sr': Sample rate
            - 'duration': Audio duration in seconds

    Example:
        >>> analysis = analyze_audio_structure('song.mp3')
        >>> print(f"Tempo: {analysis['tempo']:.1f} BPM")
        >>> print(f"Key: {analysis['key']}")
        >>> print(f"Segments: {len(analysis['segments'])}")
    """
    print(f"🎵 Analyzing audio structure: {Path(audio_fpath).name}")

    # Load audio
    y, sr = librosa.load(audio_fpath, sr=sr)
    duration = len(y) / sr

    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"  ✓ Tempo: {tempo:.1f} BPM")

    # Key detection using chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = estimate_key(chroma)
    print(f"  ✓ Key: {key}")

    # Constant-Q transform for structure
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length)
    cqt_mag = np.abs(cqt)

    # Spectral clustering for segmentation
    if n_segments is None:
        # Heuristic: ~1 segment per 8 seconds
        n_segments = max(2, int(duration / 8))

    segments = detect_segments(cqt_mag, beats, n_segments=n_segments)
    print(f"  ✓ Detected {len(segments)} structural segments")
    print(f"  ✓ Duration: {duration:.1f}s")

    return {
        'tempo': float(tempo),
        'key': key,
        'segments': segments,
        'beats': beats,
        'cqt': cqt_mag,
        'sr': sr,
        'duration': duration,
        'hop_length': hop_length
    }


def estimate_key(chroma: np.ndarray) -> str:
    """
    Estimate musical key from chromagram using template matching.

    Args:
        chroma: Chromagram matrix (12 x frames)

    Returns:
        Estimated key (e.g., 'C major', 'A minor')
    """
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Average chroma across time
    chroma_mean = np.mean(chroma, axis=1)

    # Normalize
    chroma_mean = chroma_mean / np.sum(chroma_mean)

    # Test all rotations against major and minor profiles
    best_corr = -1
    best_key = 'C major'

    for shift in range(12):
        # Rotate chroma
        chroma_rotated = np.roll(chroma_mean, shift)

        # Test major
        corr_major = np.corrcoef(chroma_rotated, major_profile)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{note_names[shift]} major"

        # Test minor
        corr_minor = np.corrcoef(chroma_rotated, minor_profile)[0, 1]
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{note_names[shift]} minor"

    return best_key


def detect_segments(cqt: np.ndarray,
                   beats: np.ndarray,
                   n_segments: int = 8) -> List[Dict]:
    """
    Detect structural segments using spectral clustering.

    Args:
        cqt: Constant-Q transform magnitude
        beats: Beat frame indices
        n_segments: Number of segments to detect

    Returns:
        List of segment dicts with 'start_frame', 'end_frame', 'label'
    """
    # Compute self-similarity matrix
    # Use librosa's recurrence matrix for efficient computation
    rec_matrix = librosa.segment.recurrence_matrix(
        cqt,
        mode='affinity',
        metric='cosine',
        bandwidth=3
    )

    # Apply spectral clustering
    n_clusters = min(n_segments, len(beats) // 4) if len(beats) > 0 else n_segments
    n_clusters = max(2, n_clusters)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42
    )

    # Sample at beat positions for efficiency
    if len(beats) > 0:
        rec_sampled = rec_matrix[:, beats][beats, :]
        labels = clustering.fit_predict(rec_sampled)

        # Map back to full timeline
        full_labels = np.zeros(cqt.shape[1], dtype=int)
        for i, beat_idx in enumerate(beats):
            if i < len(beats) - 1:
                start = beats[i]
                end = beats[i + 1]
            else:
                start = beats[i]
                end = cqt.shape[1]
            full_labels[start:end] = labels[i]
    else:
        # No beats detected, use uniform sampling
        full_labels = clustering.fit_predict(rec_matrix)

    # Convert to segments
    segments = []
    current_label = full_labels[0]
    start_frame = 0

    for i, label in enumerate(full_labels):
        if label != current_label:
            segments.append({
                'start_frame': int(start_frame),
                'end_frame': int(i),
                'label': int(current_label)
            })
            current_label = label
            start_frame = i

    # Final segment
    segments.append({
        'start_frame': int(start_frame),
        'end_frame': int(len(full_labels)),
        'label': int(current_label)
    })

    return segments


def extract_audio_features(audio_fpath: str, sr: int = 22050) -> Dict:
    """
    Extract comprehensive audio features for analysis.

    Args:
        audio_fpath: Path to audio file
        sr: Sample rate

    Returns:
        dict with tempo, key, energy, spectral features, etc.
    """
    y, sr = librosa.load(audio_fpath, sr=sr)

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Energy
    rms = librosa.feature.rms(y=y)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    return {
        'tempo': float(tempo),
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_centroid_std': float(np.std(spectral_centroid)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'zcr_mean': float(np.mean(zcr)),
        'mfcc_mean': mfcc.mean(axis=1).tolist(),
        'chroma_mean': chroma.mean(axis=1).tolist(),
        'duration': len(y) / sr
    }


# ============================================================================
# Stem Separation
# ============================================================================

def ensure_stems_separated(audio_fpath: str,
                          stems_path: Optional[Path] = None,
                          model: str = 'htdemucs_ft',
                          force: bool = False) -> Dict[str, Path]:
    """
    Separate audio into stems (vocals, drums, bass, other) using demucs.

    Uses Facebook's Demucs model for high-quality source separation.

    Args:
        audio_fpath: Path to audio file
        stems_path: Output directory (default: audio_dir/stems)
        model: Demucs model name ('htdemucs', 'htdemucs_ft', etc.)
        force: Re-separate even if stems exist

    Returns:
        dict mapping stem names to file paths:
            {'vocals': Path, 'drums': Path, 'bass': Path, 'other': Path}

    Example:
        >>> stems = ensure_stems_separated('song.mp3')
        >>> vocals_path = stems['vocals']
        >>> drums_path = stems['drums']

    Raises:
        RuntimeError: If demucs command fails
    """
    audio_path = Path(audio_fpath)
    stems_path = stems_path or audio_path.parent / 'stems'

    stem_dir = stems_path / model / audio_path.stem
    expected_stems = {
        'vocals': stem_dir / 'vocals.wav',
        'drums': stem_dir / 'drums.wav',
        'bass': stem_dir / 'bass.wav',
        'other': stem_dir / 'other.wav'
    }

    # Check if already separated
    if not force and all(p.exists() for p in expected_stems.values()):
        print(f"✓ Stems already exist: {stem_dir}")
        return expected_stems

    # Run demucs
    print(f"🎼 Separating stems with {model}...")
    print(f"   Audio: {audio_path.name}")
    print(f"   This may take 2-5 minutes depending on audio length...")

    cmd = [
        'demucs',
        '-n', model,
        '-o', str(stems_path),
        str(audio_fpath)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr}")

    print(f"  ✓ Stems saved to {stem_dir}")
    return expected_stems


def load_stem(stem_fpath: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio stem with specified sample rate.

    Args:
        stem_fpath: Path to stem audio file
        sr: Target sample rate

    Returns:
        (audio_data, sample_rate)
    """
    y, sr = librosa.load(stem_fpath, sr=sr)
    return y, sr


def get_stem_signal(stem_fpath: str,
                   sr: int = 22050,
                   hop_length: int = 512) -> np.ndarray:
    """
    Load stem and compute envelope for audioreactive effects.

    Args:
        stem_fpath: Path to stem audio file
        sr: Sample rate
        hop_length: Hop length for envelope computation

    Returns:
        Envelope signal (amplitude over time)
    """
    y, sr = librosa.load(stem_fpath, sr=sr)

    # Compute RMS energy envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    return rms


# ============================================================================
# Utility Functions
# ============================================================================

def frames_to_time(frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Convert frame indices to time in seconds."""
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


def time_to_frames(times: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Convert time in seconds to frame indices."""
    return librosa.time_to_frames(times, sr=sr, hop_length=hop_length)


def get_audio_duration(audio_fpath: str) -> float:
    """
    Get audio duration in seconds without loading full file.

    Args:
        audio_fpath: Path to audio file

    Returns:
        Duration in seconds
    """
    duration = librosa.get_duration(path=audio_fpath)
    return duration
