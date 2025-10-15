import numpy as np

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from typing import Tuple


model_name = "microsoft/wavlm-base-plus"

SR_HANDCRAFT = 16000
SR = 16000 
WAVLM_MODEL = "microsoft/wavlm-base-plus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- WavLM model loader (singleton) ----------
_wavlm_model = None
_wavlm_feature_extractor = None

def load_wavlm_model(model_name: str = WAVLM_MODEL):
    global _wavlm_model, _wavlm_feature_extractor
    if _wavlm_model is None:
        print(f"Loading WavLM model '{model_name}' to device {DEVICE} (this can take a while)...")
        _wavlm_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        _wavlm_model = AutoModel.from_pretrained(model_name).to(DEVICE)
        _wavlm_model.eval()
    return _wavlm_feature_extractor, _wavlm_model


def extract_wavlm_embedding_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Given waveform array y (float32), sample rate sr, returns a 1D numpy array embedding
    by meanpooling the model's last_hidden_state across time.
    """
    feat_extractor, model = load_wavlm_model()
    # Ensure sampling rate expected
    if sr != feat_extractor.sampling_rate:
        # Resample audio for model if needed
        y = librosa.resample(y, orig_sr=sr, target_sr=feat_extractor.sampling_rate)
        sr = feat_extractor.sampling_rate

    # feature extractor expects list of arrays
    inputs = feat_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)  
    attention_mask = inputs.get("attention_mask", None)
    with torch.no_grad():
        if attention_mask is not None:
            outputs = model(input_values, attention_mask=attention_mask)
        else:
            outputs = model(input_values)
        last_hidden = outputs.last_hidden_state  
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  
            masked_hidden = last_hidden * mask
            summed = masked_hidden.sum(dim=1)          
            denom = mask.sum(dim=1).clamp(min=1e-9)  
            pooled = summed / denom
        else:
            pooled = last_hidden.mean(dim=1) 

        emb = pooled.squeeze(0).cpu().numpy()
    return emb 

def estimate_mean_pitch(y: np.ndarray, sr: int) -> float:
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        if f0 is None:
            return 0.0
        if voiced_flag is None:
            voiced_mask = ~np.isnan(f0)
        else:
            voiced_mask = voiced_flag.astype(bool)
        f0[~voiced_mask] = np.nan
        m = float(np.nanmean(f0))
        return 0.0 if np.isnan(m) else m
    except Exception:
        return 0.0


def extract_mfccs(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # shape (n_mfcc,)
    return mfccs_mean

def extract_spectral_features(y: np.ndarray, sr: int) -> Tuple[float, float, float]:
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    sctr = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    sflat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    return sc, sctr, sflat

def extract_loudness(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))

def extract_rhythm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception:
        return 0.0


def generate_features(file_path):
    """
    Generates a concatenated feature vector for the given audio file.

    Args:
        file_path (str): Path to the input audio file.

    Returns:
        np.ndarray: Combined feature vector.
    """
    y, sr = librosa.load(file_path, sr=SR, mono=True)  # we use SR for handcrafted; extractor will resample if needed
    if y.size == 0:
        # If model not loaded, determine wavlm dim from model config if possible, else return zeros
        feat_extractor, model = load_wavlm_model()
        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0
        return np.zeros(19 + hidden_size, dtype=np.float32)

    # handcrafted
    mean_pitch = estimate_mean_pitch(y, sr)
    mfcc_mean = extract_mfccs(y, sr, n_mfcc=13)  # (13,)
    sc, sctr, sflat = extract_spectral_features(y, sr)
    loudness = extract_loudness(y)
    tempo = extract_rhythm(y, sr)

    handcrafted = np.concatenate(
        (
            np.array([mean_pitch], dtype=np.float32),
            mfcc_mean.astype(np.float32),
            np.array([sc, sctr, sflat, loudness, tempo], dtype=np.float32),
        )
    )  # shape (19,)

    # wavlm embedding
    try:
        wavlm_emb = extract_wavlm_embedding_from_audio(y, sr)  # 1D numpy
    except Exception as e:
        print(f"Warning: WavLM embedding failed for {file_path}: {e}")
        # fall back to zeros of appropriate dim
        feat_extractor, model = load_wavlm_model()
        hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 0
        wavlm_emb = np.zeros(hidden_size, dtype=np.float32)

    # final vector
    final = np.concatenate([handcrafted.astype(np.float32), wavlm_emb.astype(np.float32)])

    return final