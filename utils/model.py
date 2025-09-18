import torch
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import soundfile as sf

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def extend_audio_and_save(y, sr, target_duration):
    """
    Extends an audio clip to a target duration by repeating (tiling) it.
    This is our base method for extending duration-preserving augmentations.
    """
    target_samples = int(target_duration * sr)
    num_repeats = int(np.ceil(target_samples / len(y)))
    extended_audio = np.tile(y, num_repeats)
    extended_audio = extended_audio[:target_samples]

    # save .wav file
    sf.write("extended_audio.wav", extended_audio, sr)
    return extended_audio

def extract_embedding(waveform, sr=16000):
    """
    Extracts a robust, normalized embedding from the entire audio waveform at once.
    """
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)

    # Resample to the model's expected sample rate (16000 Hz) if needed
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    # Pass the entire waveform to the feature extractor.
    # The `padding=True` argument handles waveforms of different lengths.
    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

        # Average the last 4 hidden layers for a more robust representation
        # Squeeze removes the batch dimension (we process one file at a time)
        hidden_states = torch.stack(outputs.hidden_states[-4:]).mean(0).squeeze(0)

        # Apply temporal pooling (mean + std) over the sequence of hidden states
        mean = hidden_states.mean(dim=0)
        std = hidden_states.std(dim=0)
        pooled_embedding = torch.cat([mean, std])

    # Normalize the final embedding vector
    final_embedding = F.normalize(pooled_embedding, p=2, dim=0)

    return final_embedding.cpu()

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
