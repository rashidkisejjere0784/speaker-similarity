import streamlit as st
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import io
import time
from utils.model import extract_embedding, cosine_similarity, extend_audio_and_save
import torch

# Password check function
def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"] # don't store the password
        else:
            st.session_state["password_correct"] = False
    
    # Check if the password is correct in session state
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error("Password incorrect")
    return False

if not check_password():
    st.stop()


# Page Configuration
st.set_page_config(
    page_title="Audio Similarity Configurator",
    page_icon="🎵",
    layout="wide"
)

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
        wav, sr = librosa.load(audio_stream, sr=None)

        # extend audio to 5 seconds and save
        if len(wav) < sr * 5:
            wav = extend_audio_and_save(wav, sr, target_duration=5.0)

        wav = torch.tensor(wav)
        # Extract speaker embedding
        embedding = extract_embedding(wav)
        return embedding
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Error extracting features: {e}")
        return None

# Initialize session state variables
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
    st.session_state.calculation_time_ms = None

#  UI Sidebar
st.sidebar.header("Configuration")

# Base Audio uploader
st.sidebar.subheader("1. Upload Base Audio")
base_audio_file = st.sidebar.file_uploader(
    "This is the reference audio.", type=["wav", "mp3", "flac", "m4a"]
)

# Process the base audio file when uploaded
if base_audio_file is not None:
    if base_audio_file.name != st.session_state.base_audio_name:
        st.session_state.base_audio_name = base_audio_file.name
        st.session_state.base_audio_bytes = base_audio_file.getvalue()
        with st.spinner("Analyzing base audio..."):
            st.session_state.base_audio_features = extract_features(st.session_state.base_audio_bytes)
        # Reset comparison state when a new base audio is uploaded
        st.session_state.comparison_audio_features = None
        st.session_state.comparison_audio_name = None
        st.session_state.similarity_score = None
        st.session_state.calculation_time_ms = None
        st.sidebar.success(f"Base audio '{st.session_state.base_audio_name}' loaded.")

# Comparison Audio Uploader only if base is loaded
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

                # ADD TIMER
                start_time = time.perf_counter()
                # Cosine Similarity calculation
                score = cosine_similarity(base_feat, comp_feat)
                end_time = time.perf_counter()
                # END TIMER

                st.session_state.similarity_score = score
                st.session_state.calculation_time_ms = (end_time - start_time) * 1000
            else:
                st.session_state.similarity_score = None
                st.session_state.calculation_time_ms = None

# Threshold Slider
st.sidebar.subheader("3. Adjust Similarity Threshold")
threshold = st.sidebar.slider(
    "Set the threshold for similarity (0 to 1)",
    min_value=0.0,
    max_value=1.0,
    value=0.57,
    step=0.001
)

#  Main Application Body
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

#  Results Section
st.header("Results")
if st.session_state.similarity_score is not None:
    score = st.session_state.similarity_score
    calc_time = st.session_state.calculation_time_ms

    # Create two columns for the metrics
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Cosine Similarity Score", value=f"{score:.4f}")
    with res_col2:
        # DISPLAY TIMER
        st.metric(label="Calculation Time", value=f"{calc_time:.2f} ms")


    # Compare with the threshold and display the result
    if score >= threshold:
        st.success(f"Similar (Score ≥ {threshold})")
    else:
        st.error(f"Not Similar (Score < {threshold})")

elif st.session_state.base_audio_name and not st.session_state.comparison_audio_name:
    st.info("Now upload a comparison audio to see the similarity score.")
else:
    st.info("Upload both a base and a comparison audio to begin.")