import librosa
import numpy as np


def extract_enhanced_features(segment, sr=16000):
    """Extract multiple audio features for better classification"""
    
    # Original MFCC features
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=64, n_fft=512*2, hop_length=int(sr * 0.010))
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
    
    # Zero crossing rate (good for speech vs non-speech)
    zcr = librosa.feature.zero_crossing_rate(segment)[0]
    
    # Chroma features (good for harmonic content)
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    
    # Tempo and rhythm (good for footsteps, rolling carts)
    tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)

    if type(tempo) is float:
        tempo = [tempo]
    
    # Combine all features - ensure all are 1D arrays
    features = np.concatenate([
        mfcc_mean,  # Shape: (13,)
        mfcc_std,   # Shape: (13,)
        np.array([spectral_centroids.mean(), spectral_centroids.std()]),  # Shape: (2,)
        np.array([spectral_rolloff.mean(), spectral_rolloff.std()]),      # Shape: (2,)
        np.array([spectral_bandwidth.mean(), spectral_bandwidth.std()]),  # Shape: (2,)
        np.array([zcr.mean(), zcr.std()]),                                # Shape: (2,)
        chroma_mean,                                                      # Shape: (12,)
        np.array([float(tempo[0]) if len(tempo) > 0 else 0.0, len(beats)])  # Shape: (2,)
    ])
    return features, mfcc