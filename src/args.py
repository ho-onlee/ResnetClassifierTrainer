import random
import torch
import torchaudio
import librosa
import numpy as np
from dynaconf import Dynaconf
import numpy.typing as npt

settings = Dynaconf(settings_files=["../settings.toml", "../settings.local.toml"])

def random_gain(x, db_range=settings.args.gain):
    gain = random.uniform(*db_range)
    factor = 10 ** (gain / 20)
    return x * factor

def random_noise(x, noise_level=settings.args.noise_level):
    noise_amp = random.uniform(*noise_level)
    noise = torch.randn_like(x) * noise_amp
    return x + noise

def random_lowpass(x, cutoff_range=settings.args.lowpass):
    cutoff = random.uniform(*cutoff_range)
    return torchaudio.functional.lowpass_biquad(torch.from_numpy(x), settings.audio.sr, cutoff)

def random_highpass(x, cutoff_range=settings.args.highpass):
    cutoff = random.uniform(*cutoff_range)
    return torchaudio.functional.highpass_biquad(torch.from_numpy(x), settings.audio.sr, cutoff)

def random_speed(x, speed_range=settings.args.speed):
    rate = random.uniform(*speed_range)
    return torchaudio.functional.resample(torch.from_numpy(x), settings.audio.sr, int(settings.audio.sr * rate))

def random_pitch(x, pitch_range=settings.args.pitch):
    shift = random.randint(*pitch_range) 
    return torchaudio.functional.pitch_shift(torch.from_numpy(x), settings.audio.sr, shift)


def add_noise(segment, noise_factor: float = 0.005) -> npt.NDArray[np.float32]:
    """
    Add Gaussian noise to audio segment for data augmentation.
    
    Input:
        segment (np.ndarray): Audio segment to augment
        noise_factor (float): Strength of noise to add
        
    Output:
        np.ndarray: Audio segment with added noise
    """
    noise = np.random.randn(len(segment))
    augmented_segment = segment + noise_factor * noise
    return augmented_segment

def add_reverb(segment, reverberation_factor: float = 0.5) -> npt.NDArray[np.float32]:
    """
    Add reverb effect to audio segment using convolution.
    
    Input:
        segment (np.ndarray): Audio segment to process
        reverberation_factor (float): Strength of reverb effect
        
    Output:
        np.ndarray: Audio segment with reverb effect applied
    """
    # Ensure we always have at least 1 sample for the impulse response
    try:
        impulse_length = max(1, int(reverberation_factor * 1000))
        impulse_response = np.ones(impulse_length) / impulse_length
        reverb = np.convolve(segment, impulse_response, mode='same')
    except Exception as e:
        print(f"Error adding reverb: {e}, segment length: {len(segment)}")
        exit()
    return reverb

def add_stretch(segment, rate: float = 1.1) -> npt.NDArray[np.float32]:
    """
    Apply time stretching to audio segment for data augmentation.
    
    Input:
        segment (np.ndarray): Audio segment to stretch
        rate (float): Stretch rate (>1.0 makes audio faster, <1.0 slower)
        
    Output:
        np.ndarray: Time-stretched audio segment
    """
    stretched = librosa.effects.time_stretch(segment, rate=rate)
    return stretched

def add_pitch_shift(segment, sr: int = settings.audio.sr, n_steps: int = 2) -> npt.NDArray[np.float32]:
    """
    Apply pitch shifting to audio segment for data augmentation.
    
    Input:
        segment (np.ndarray): Audio segment to pitch shift
        sr (int, optional): Sample rate. Uses self.sr if None
        n_steps (int): Number of semitones to shift (positive = higher pitch)
        
    Output:
        np.ndarray: Pitch-shifted audio segment
    """
    pitched = librosa.effects.pitch_shift(segment, sr=sr, n_steps=n_steps)
    return pitched