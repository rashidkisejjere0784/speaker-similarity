import torch
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()


def extract_embedding(waveform, sr=16000, chunk_size=0.5, overlap=0.25):
    """Extract robust normalized embedding from audio waveform."""
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)

    # Resample if needed
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunk_len = int(chunk_size * sr)
    step = int((chunk_size - overlap) * sr)
    step = max(1, step) 

    if len(waveform) < chunk_len:
        waveform = np.pad(waveform, (0, chunk_len - len(waveform)), mode="constant")

    embeddings = []
    for start in range(0, len(waveform) - chunk_len + 1, step):
        chunk = waveform[start:start + chunk_len]

        inputs = feature_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

            # Average the last 4 layers for more robust embeddings
            hidden_states = torch.stack(outputs.hidden_states[-4:]).mean(0).squeeze(0)

            # Pooling: mean + std
            mean = hidden_states.mean(dim=0)
            std = hidden_states.std(dim=0)
            pooled = torch.cat([mean, std])

            embeddings.append(pooled.cpu())

    # Aggregate across chunks
    final_embedding = torch.stack(embeddings).mean(dim=0)
    final_embedding = F.normalize(final_embedding, dim=0)

    return final_embedding


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
