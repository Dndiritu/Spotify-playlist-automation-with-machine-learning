import numpy as np
import pandas as pd
import streamlit as st
import joblib

#Loading saved components
model = joblib.load('kmeans_cluster_model.pkl')
scaler = joblib.load('scalerr.pkl')
pca = joblib.load('pca.pkl')
label_map = joblib.load("label_map.pkl")

st.title('Spotify Songs Playlist Automation')
st.write('This app analyzes your Spotify playlists and classifies songs into various genres or categories. By leveraging machine learning algorithms, it helps you understand the composition of your playlists, discover new music trends, and organize your songs more effectively. Upload your playlist data or connect your Spotify account to get personalized insights and classifications.')


st.markdown("### 🎵 Input Song Features")

popularity = st.sidebar.number_input("Popularity (Range: -1.48 to 2.74)", min_value=-1.48, max_value=2.74, value=0.0, step=0.01)
danceability = st.sidebar.number_input("Danceability (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
energy = st.sidebar.number_input("Energy (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
loudness = st.sidebar.number_input("Loudness (-60.0 – 0.0)", min_value=-60.0, max_value=0.0, value=-12.0, step=1.0)
speechiness = st.sidebar.number_input("Speechiness (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
acousticness = st.sidebar.number_input("Acousticness (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
instrumentalness = st.sidebar.number_input("Instrumentalness (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
liveness = st.sidebar.number_input("Liveness (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
valence = st.sidebar.number_input("Valence (0.0 – 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
tempo = st.sidebar.number_input("Tempo (50.0 – 200.0)", min_value=50.0, max_value=200.0, value=120.0, step=1.0)
duration_ms = st.sidebar.number_input("Duration (milliseconds)", min_value=50000, max_value=500000, value=180000, step=1000)
key = st.sidebar.number_input("Key (0 to 11)", min_value=0, max_value=11, value=5, step=1)
mode = st.sidebar.number_input("Mode (0 = minor, 1 = major)", min_value=0, max_value=1, value=1, step=1)
time_signature = st.sidebar.number_input("Time Signature (typically 3 – 7)", min_value=3, max_value=7, value=4, step=1)
explicit = st.sidebar.number_input("Explicit (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0, step=1)

# ✅ Create feature list
song_features = [
    popularity,
    acousticness,
    danceability,
    duration_ms,
    energy,
    instrumentalness,
    key,
    liveness,
    loudness,
    mode,
    speechiness,
    tempo,
    time_signature,
    valence,
    explicit
]


if st.button("🔍 Predict Cluster"):
    scaled = scaler.transform([song_features])
    reduced = pca.transform(scaled)
    cluster = model.predict(reduced)[0]
    label = label_map.get(cluster, "Unknown Cluster")

    st.success(f"🎧 This song belongs to **Cluster {cluster}**: *{label}*")
