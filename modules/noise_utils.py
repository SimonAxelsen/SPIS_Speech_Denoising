"""
Noise generation and audio mixing utilities
"""
import numpy as np
from scipy import signal


def generate_white_noise(duration, sample_rate):
    """Generate white noise (uniform power across frequencies)"""
    num_samples = int(duration * sample_rate)
    return np.random.randn(num_samples)


def generate_pink_noise(duration, sample_rate):
    """Generate pink noise (1/f power spectrum)"""
    num_samples = int(duration * sample_rate)
    
    # Generate white noise
    white = np.random.randn(num_samples)
    
    # Apply 1/f filter using FFT
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, 1/sample_rate)
    
    # Avoid division by zero
    freqs[0] = 1e-10
    
    # Apply 1/f scaling
    pink_fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=num_samples)
    
    # Normalize
    pink = pink / np.max(np.abs(pink))
    
    return pink


def generate_babble_noise(duration, sample_rate):
    """Generate babble noise (simulated crowd/cafeteria noise)"""
    num_samples = int(duration * sample_rate)
    
    # Mix multiple modulated noise sources to simulate voices
    babble = np.zeros(num_samples)
    
    num_voices = 5  # Simulate 5 background speakers
    for i in range(num_voices):
        # Random frequency modulation to simulate speech
        freq = np.random.uniform(80, 300)  # Hz, typical speech fundamental
        t = np.arange(num_samples) / sample_rate
        
        # Amplitude modulation (syllabic rate)
        mod_freq = np.random.uniform(2, 5)  # Hz
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2*np.pi))
        
        # Carrier with formant-like filtering
        carrier = np.random.randn(num_samples)
        
        # Simple formant filtering (2-3 resonances)
        sos = signal.butter(4, [freq, freq*3], 'bandpass', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, carrier)
        
        babble += filtered * modulation
    
    # Normalize
    babble = babble / np.max(np.abs(babble) + 1e-10)
    
    return babble


def calculate_rms(audio):
    """Calculate RMS (Root Mean Square) energy of audio signal"""
    return np.sqrt(np.mean(audio ** 2))


def calculate_snr(clean, noisy):
    """Calculate Signal-to-Noise Ratio in dB"""
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((noisy - clean) ** 2)
    
    if noise_power < 1e-10:
        return 100.0  # Effectively infinite SNR
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def add_noise_at_snr(clean_audio, noise_audio, target_snr_db):
    """
    Mix clean audio with noise at a specific SNR level
    
    Args:
        clean_audio: Clean signal
        noise_audio: Noise signal (will be truncated/repeated to match length)
        target_snr_db: Target SNR in decibels
    
    Returns:
        Noisy audio at target SNR
    """
    # Ensure noise is same length as clean audio
    if len(noise_audio) < len(clean_audio):
        # Repeat noise if too short
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)
    
    noise_audio = noise_audio[:len(clean_audio)]
    
    # Calculate RMS of clean signal and noise
    rms_clean = calculate_rms(clean_audio)
    rms_noise = calculate_rms(noise_audio)
    
    # Calculate required noise scaling factor
    # SNR = 20*log10(rms_signal / rms_noise)
    # => rms_noise_target = rms_signal / 10^(SNR/20)
    snr_linear = 10 ** (target_snr_db / 20.0)
    rms_noise_target = rms_clean / snr_linear
    
    # Scale noise
    if rms_noise > 1e-10:
        noise_scaled = noise_audio * (rms_noise_target / rms_noise)
    else:
        noise_scaled = noise_audio
    
    # Mix
    noisy_audio = clean_audio + noise_scaled
    
    # Prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
    
    return noisy_audio


def generate_noise_by_type(noise_type, duration, sample_rate):
    """
    Generate noise based on type
    
    Args:
        noise_type: "white", "pink", "babble", or "none"
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Noise array
    """
    if noise_type.lower() == "white":
        return generate_white_noise(duration, sample_rate)
    elif noise_type.lower() == "pink":
        return generate_pink_noise(duration, sample_rate)
    elif noise_type.lower() == "babble":
        return generate_babble_noise(duration, sample_rate)
    elif noise_type.lower() == "none":
        return np.zeros(int(duration * sample_rate))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
