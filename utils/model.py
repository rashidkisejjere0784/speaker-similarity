import torch
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def extract_embedding(waveform, sr=16000, chunk_size=0.5, overlap=0.25):
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunk_len = int(chunk_size * sr)
    step = int((chunk_size - overlap) * sr)

    if len(waveform) < chunk_len:
        pad = chunk_len - len(waveform)
        waveform = np.pad(waveform, (0, pad), mode="constant")

    embeddings = []
    for start in range(0, max(1, len(waveform) - chunk_len + 1), step):
        chunk = waveform[start:start + chunk_len]

        inputs = feature_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.squeeze(0)

            mean = hidden_states.mean(dim=0)
            std = hidden_states.std(dim=0)
            pooled = torch.cat([mean, std])

            embeddings.append(pooled.cpu())

    final_embedding = torch.stack(embeddings).mean(dim=0)
    final_embedding = F.normalize(final_embedding, dim=0)

    return final_embedding

def cosine_similarity(emb1, emb2):
    emb1 = emb1.unsqueeze(0)
    emb2 = emb2.unsqueeze(0)
    return torch.mm(emb1, emb2.T).item()