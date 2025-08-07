import streamlit as st
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine

# Page Configuration
st.set_page_config(
    page_title="Audio Similarity Configurator",
    page_icon="🎵",
    layout="wide"
)

encoder = VoiceEncoder()

@st.cache_data
def extract_features(audio_bytes):
    """
    Extracts speaker embeddings from an audio file byte stream using Resemblyzer.

    Args:
        audio_bytes (bytes): The byte content of the audio file.

    Returns:
        np.ndarray: An embedding vector representing the speaker.
    """
    try:
        # Use a file-like object for librosa
        audio_stream = io.BytesIO(audio_bytes)

        # Load the waveform using librosa (returns audio and sampling rate)
        wav, sr = librosa.load(audio_stream, sr=None)  # keep original sample rate

        # Convert to Resemblyzer-compatible format
        wav = preprocess_wav(wav, source_sr=sr)

        # Extract speaker embedding
        embedding = encoder.embed_utterance(wav)

        return embedding

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error extracting features: {e}")
        return None

if 'base_audio_features' not in st.session_state:
    st.session_state.base_audio_features = None
    st.session_state.base_audio_name = None
    st.session_state.base_audio_bytes = None

if 'comparison_audio_features' not in st.session_state:
    st.session_state.comparison_audio_features = None
    st.session_state.comparison_audio_name = None
    st.session_state.comparison_audio_bytes = None
    
if 'similarity_score' not in st.session_state:
    st.session_state.similarity_score = None

# --- UI Sidebar ---
st.sidebar.header("Configuration")

# 1. Base Audio Uploader
st.sidebar.subheader("1. Upload Base Audio")
base_audio_file = st.sidebar.file_uploader(
    "This is the reference audio.", type=["wav", "mp3", "flac", "m4a"]
)

# Process the base audio file when uploaded
if base_audio_file is not None:
    # Check if a new file has been uploaded
    if base_audio_file.name != st.session_state.base_audio_name:
        st.session_state.base_audio_name = base_audio_file.name
        st.session_state.base_audio_bytes = base_audio_file.getvalue()
        with st.spinner("Analyzing base audio..."):
            st.session_state.base_audio_features = extract_features(st.session_state.base_audio_bytes)
        # Reset comparison when base changes
        st.session_state.comparison_audio_features = None
        st.session_state.similarity_score = None
        st.sidebar.success(f"Base audio '{st.session_state.base_audio_name}' loaded.")

# 2. Comparison Audio Uploader only if base is loaded
if st.session_state.base_audio_features is not None:
    st.sidebar.subheader("2. Upload Comparison Audio")
    comparison_audio_file = st.sidebar.file_uploader(
        "Upload audio to compare against the base.", type=["wav", "mp3", "flac", "m4a"]
    )

    # Process the comparison audio file when uploaded
    if comparison_audio_file is not None:
        if comparison_audio_file.name != st.session_state.comparison_audio_name:
            st.session_state.comparison_audio_name = comparison_audio_file.name
            st.session_state.comparison_audio_bytes = comparison_audio_file.getvalue()
            with st.spinner("Analyzing comparison audio..."):
                st.session_state.comparison_audio_features = extract_features(st.session_state.comparison_audio_bytes)

            # Calculate similarity if both features are available
            if st.session_state.comparison_audio_features is not None:
                base_feat = st.session_state.base_audio_features
                comp_feat = st.session_state.comparison_audio_features
                score = 1 - cosine(base_feat, comp_feat)  # Cosine similarity
                st.session_state.similarity_score = score
            else:
                st.session_state.similarity_score = None

# 3. Threshold Slider
st.sidebar.subheader("3. Adjust Similarity Threshold")
threshold = st.sidebar.slider(
    "Set the threshold for similarity (0 to 1)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.90,
    step=0.01
)

# --- Main Application Body ---
st.title("🎵 Audio Similarity Configurator")
st.markdown(
    "This tool helps you find the optimal **similarity threshold** between two audio files. "
    "Start by uploading a base audio, then upload a comparison audio. Adjust the slider to see the result."
)

st.divider()

col1, col2 = st.columns(2)

# Display Base Audio
with col1:
    st.header("Base Audio")
    if st.session_state.base_audio_name:
        st.info(f"**File:** `{st.session_state.base_audio_name}`")
        st.audio(st.session_state.base_audio_bytes)
    else:
        st.warning("Please upload a base audio file from the sidebar.")

# Display Comparison Audio
with col2:
    st.header("Comparison Audio")
    if st.session_state.comparison_audio_name and st.session_state.base_audio_name:
        st.info(f"**File:** `{st.session_state.comparison_audio_name}`")
        st.audio(st.session_state.comparison_audio_bytes)
    else:
        st.warning("Please upload a comparison audio file.")

st.divider()

# --- Results Section ---
st.header("Results")
if st.session_state.similarity_score is not None:
    score = st.session_state.similarity_score
    
    # Display the metric
    st.metric(label="Cosine Similarity Score", value=f"{score:.4f}")

    # Compare with the threshold and display the result
    if score >= threshold:
        st.success(f"Similar (Score ≥ {threshold})")
    else:
        st.error(f"Not Similar (Score < {threshold})")
    
elif st.session_state.base_audio_name and not st.session_state.comparison_audio_name:
    st.info("Now upload a comparison audio to see the similarity score.")
else:
    st.info("Upload both a base and a comparison audio to begin.")