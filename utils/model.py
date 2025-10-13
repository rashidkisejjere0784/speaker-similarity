import torch
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import soundfile as sf
from helper.extract_features import generate_features
from helper.ml_models import get_prediction

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def extend_audio_and_save(
    y,
    sr,
    target_duration,
    output_path="extended_audio.wav",
    pitch_range=(-2, 2),
    stretch_range=(0.9, 1.1),
    noise_level=0.005
):
    """
    Extends an audio clip to a target duration by repeating (tiling) it,
    then applies pitch shift, time stretch, and noise addition while
    ensuring the final output matches the target duration.

    Works with librosa >= 0.10 (tested on 0.11.0).
    """

    # --- Step 1: Extend to target duration ---
    target_samples = int(target_duration * sr)
    num_repeats = int(np.ceil(target_samples / len(y)))
    extended_audio = np.tile(y, num_repeats)[:target_samples]

    # --- Step 2: Apply pitch shift ---
    n_steps = np.random.uniform(*pitch_range)
    pitched_audio = librosa.effects.pitch_shift(y=extended_audio, sr=sr, n_steps=n_steps)

    # --- Step 3: Apply time stretching ---
    stretch_rate = np.random.uniform(*stretch_range)
    stretched_audio = librosa.effects.time_stretch(y=pitched_audio, rate=stretch_rate)

    # --- Step 4: Match target duration again ---
    if len(stretched_audio) < target_samples:
        stretched_audio = np.tile(stretched_audio, int(np.ceil(target_samples / len(stretched_audio))))
    final_audio = stretched_audio[:target_samples]

    # --- Step 5: Add Gaussian noise ---
    noise = np.random.normal(0, noise_level, len(final_audio))
    final_audio = final_audio + noise

    # --- Step 6: Save ---
    sf.write(output_path, final_audio, sr)

    print(f"Applied augmentations: pitch shift={n_steps:.2f} semitones, "
          f"stretch rate={stretch_rate:.3f}, noise σ={noise_level}")
    print(f"Saved to: {output_path}")

    return final_audio

def extract_embedding(waveform, sr=16000):
    """
    Extracts a robust, normalized embedding from the entire audio waveform at once.
    """
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)

    # Resample to the model's expected sample rate (16000 Hz) if needed
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    # temporarily save to a file temp.wav to use the feature extractor
    sf.write("temp.wav", waveform, 16000)

    # inputs = feature_extractor(
    #     waveform,
    #     sampling_rate=16000,
    #     return_tensors="pt",
    #     padding=True
    # ).to(device)

    # with torch.inference_mode():
    #     outputs = model(**inputs)

    #     # Average the last 4 hidden layers for a more robust representation
    #     # Squeeze removes the batch dimension (we process one file at a time)
    #     hidden_states = torch.stack(outputs.hidden_states[-4:]).mean(0).squeeze(0)

    #     # Apply temporal pooling (mean + std) over the sequence of hidden states
    #     mean = hidden_states.mean(dim=0)
    #     std = hidden_states.std(dim=0)
    #     pooled_embedding = torch.cat([mean, std])

    # # Normalize the final embedding vector
    # final_embedding = F.normalize(pooled_embedding, p=2, dim=0)

    features = generate_features("temp.wav")

    return features

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    # return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    X = np.hstack((emb1, emb2))

    preds = get_prediction(X)
    return preds[0][1]  # probability of the positive class
