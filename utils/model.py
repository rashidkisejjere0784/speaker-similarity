import torch
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True).to(device)

def extract_embedding(audio, sr=16000):
    # Resample if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Tokenize input
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.inference_mode():
        # Forward pass
        outputs = model(input_values, output_hidden_states=True)

        # Last hidden state: [batch, time, dim]
        last_hidden_state = outputs.last_hidden_state.squeeze(0)  

        # Mean pooling over time dimension
        embedding = last_hidden_state.mean(dim=0)

    return embedding.cpu()


def cosine_similarity(emb1, emb2):
    """
    Calculates the cosine similarity between two embedding tensors.
    """
    emb1 = F.normalize(emb1, p=2, dim=0)
    emb2 = F.normalize(emb2, p=2, dim=0)
    return torch.dot(emb1, emb2).item()