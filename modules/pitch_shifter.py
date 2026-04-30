"""
Pitch shifting using pseudo-cepstrum method
"""
import numpy as np
import librosa
from scipy.fftpack import dct, idct


def pseudo_cepstrum_pitch_shift(audio, sr, n_steps):
    """
    Pitch shift audio using pseudo-cepstrum method
    
    Args:
        audio: Input audio array
        sr: Sample rate
        n_steps: Pitch shift in semitones (positive = higher, negative = lower)
    
    Returns:
        Pitch-shifted audio
    """
    if n_steps == 0:
        return audio
    
    n_fft = 2048
    hop_length = 512

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)

    freq_bins, n_frames = magnitude.shape
    shift_ratio = 2 ** (n_steps / 12.0)
    lifter_cutoff = 30

    new_magnitude = np.zeros_like(magnitude)

    for t in range(n_frames):
        frame_mag = magnitude[:, t]

        if np.max(frame_mag) < 1e-10:
            new_magnitude[:, t] = frame_mag
            continue

        log_mag = np.log(frame_mag + 1e-10)
        cepstrum = dct(log_mag, norm="ortho")

        envelope_cep = np.zeros_like(cepstrum)
        envelope_cep[:lifter_cutoff] = cepstrum[:lifter_cutoff]
        log_envelope = idct(envelope_cep, norm="ortho")

        log_fine = log_mag - log_envelope

        orig_indices = np.arange(freq_bins)
        source_indices = orig_indices / shift_ratio

        shifted_fine = np.interp(
            source_indices, orig_indices, log_fine,
            left=0.0, right=0.0,
        )

        new_log_mag = log_envelope + shifted_fine
        new_magnitude[:, t] = np.exp(np.clip(new_log_mag, -20, 20))

    new_magnitude = np.maximum(new_magnitude, 0)
    shifted_audio = librosa.griffinlim(
        new_magnitude, hop_length=hop_length, n_iter=32  # Reduced iterations for speed
    )
    
    expected_len = len(audio)
    if len(shifted_audio) > expected_len:
        shifted_audio = shifted_audio[:expected_len]
    elif len(shifted_audio) < expected_len:
        shifted_audio = np.pad(shifted_audio, (0, expected_len - len(shifted_audio)))
    
    return shifted_audio
