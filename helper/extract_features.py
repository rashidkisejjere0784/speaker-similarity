import numpy as np
"""
feature_generator.py
--------------------
Generates a combined feature vector from an audio file, including:
- Mean pitch (pYIN)
- MFCCs (mean over time)
- Spectral features (centroid, contrast, flatness)
- Loudness (RMS)
- Rhythm (tempo)
"""

import librosa
import numpy as np


def estimate_mean_pitch(file_path):
    """
    Estimates the mean pitch of an audio file using the pYIN algorithm.
    Returns: float (mean pitch in Hz)
    """
    y, sr = librosa.load(file_path)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    if f0 is None:
        return 0.0
    f0[voiced_flag == False] = np.nan
    mean_pitch = np.nanmean(f0)
    return mean_pitch if not np.isnan(mean_pitch) else 0.0


def extract_mfccs(file_path):
    """
    Extracts MFCC features and averages them over time.

    Returns:
        np.ndarray: 13-dimensional vector of mean MFCCs
    """
    y, sr = librosa.load(file_path, sr=16000)  # Standard sample rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean across time
    return mfccs_mean


def extract_spectral_features(file_path):
    """
    Extracts spectral centroid, contrast, and flatness.
    Returns: dict
    """
    y, sr = librosa.load(file_path)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_contrast': spectral_contrast,
        'spectral_flatness': spectral_flatness
    }


def extract_loudness(file_path):
    """
    Extracts average loudness using RMS energy.
    Returns: float
    """
    y, _ = librosa.load(file_path)
    rms = librosa.feature.rms(y=y)[0]
    return np.mean(rms)


def extract_rhythm(file_path):
    """
    Estimates the tempo (in BPM).
    Returns: float
    """
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def generate_features(file_path):
    """
    Generates a concatenated feature vector for the given audio file.

    Args:
        file_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Combined feature vector.
    """
    features = []

    # 1. Pitch
    mean_pitch = estimate_mean_pitch(file_path)
    features.append(mean_pitch)

    # 2. MFCCs
    mfccs_mean = extract_mfccs(file_path)
    features.extend(mfccs_mean.tolist())

    # 3. Spectral features
    spectral = extract_spectral_features(file_path)
    features.extend([
        spectral['spectral_centroid'],
        spectral['spectral_contrast'],
        spectral['spectral_flatness']
    ])

    # 4. Loudness
    loudness = extract_loudness(file_path)
    features.append(loudness)

    # 5. Rhythm
    tempo = extract_rhythm(file_path)
    features.append(tempo)

    # Convert to numpy array
    feature_vector = np.array(features, dtype=np.float32)
    return feature_vector
