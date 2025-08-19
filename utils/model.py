import torch
import librosa
import torch.nn.functional as F
from transformers import AutoProcessor, HubertModel

# Load the correct processor and model
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def extract_embedding(waveform):
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.squeeze(0)
        embedding = hidden_states.mean(dim=0) 
    return embedding

def cosine_similarity(emb1, emb2):
    emb1 = F.normalize(emb1.unsqueeze(0), dim=1)
    emb2 = F.normalize(emb2.unsqueeze(0), dim=1)
    return torch.mm(emb1, emb2.T).item()