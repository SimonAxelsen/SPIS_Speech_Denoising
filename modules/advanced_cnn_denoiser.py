"""
Advanced CNN-based speech denoiser incorporating Voice Activity Detection (VAD),
noise estimation, and mask-based learning (Ideal Ratio Mask), as inspired by
traditional statistical methods combined with deep learning.

Reference: https://speechprocessingbook.aalto.fi/Enhancement/Noise_attenuation.html
"""
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def estimate_noise_profile(magnitude_spectrogram):
    """
    Estimate noise profile using energy-based Voice Activity Detection (VAD).
    Assumes low-energy frames are noise-only.
    
    Args:
        magnitude_spectrogram: 2D array of shape (freq_bins, time_frames)
        
    Returns:
        noise_estimate_mag: 1D array of shape (freq_bins,)
        vad_mask: 1D boolean array of shape (time_frames,)
    """
    # 1. Compute frame energy
    frame_energy = np.sum(magnitude_spectrogram**2, axis=0) # shape: (time_frames,)
    frame_energy_dB = 10 * np.log10(frame_energy + 1e-10)
    
    # 2. Simple VAD threshold
    mean_energy_dB = np.mean(frame_energy_dB)
    noise_threshold_dB = mean_energy_dB - 3.0
    
    noise_active = frame_energy_dB < noise_threshold_dB
    
    # Fallback if no noise frames detected
    if not np.any(noise_active):
        noise_active = frame_energy_dB < mean_energy_dB
        
    # 3. Estimate noise from inactive frames
    if not np.any(noise_active):
        # Extreme fallback to minimum statistics
        noise_estimate_energy = np.min(magnitude_spectrogram**2, axis=1)
    else:
        noise_frames = magnitude_spectrogram[:, noise_active]
        noise_estimate_energy = np.mean(noise_frames**2, axis=1)
        
    noise_estimate_mag = np.sqrt(noise_estimate_energy)
    
    return noise_estimate_mag, ~noise_active


def build_advanced_denoiser_model(input_shape=(513, 200, 3)):
    """
    Build a CNN autoencoder that predicts a spectral mask.
    
    Inputs (3 channels):
      1. Noisy Spectrogram Log-Magnitude
      2. Noise Profile Estimate Log-Magnitude (broadcasted to time dimension)
      3. VAD Feature (broadcasted to frequency dimension)
      
    Output (1 channel):
      1. Spectral Mask (sigmoid activation, values 0 to 1)
      
    Args:
        input_shape: (freq_bins, time_frames, channels)
    
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Output conv: Predict mask in [0, 1]
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='mask_output')(x)
    
    # Crop to match input: 516 -> 513
    freq_bins = input_shape[0]
    if freq_bins == 513:
        x = layers.Cropping2D(cropping=((0, 3), (0, 0)))(x)
    
    model = keras.Model(inputs, x, name='advanced_mask_denoiser')
    return model


class AdvancedAudioDenoiser:
    """Wrapper for advanced audio denoising using CNN with VAD and masking"""
    
    def __init__(self, model_path=None, sr=22050, n_fft=1024, hop_length=256):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = None
        
        if model_path and tf.io.gfile.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                print(f"Loaded advanced denoiser model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Will use untrained basic model")
        
        if self.model is None:
            print("Creating new untrained advanced model...")
            self.model = build_advanced_denoiser_model()
            
    def prepare_features(self, audio):
        """Extract multi-channel features for the advanced model."""
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Log magnitude
        log_mag = np.log(magnitude + 1e-8)
        
        # Estimate noise and VAD
        noise_mag, vad_active = estimate_noise_profile(magnitude)
        log_noise_mag = np.log(noise_mag + 1e-8)
        
        # Create broadcasted channels
        # Channel 1: Noisy Log-Mag
        # Channel 2: Noise Estimate Log-Mag (broadcasted to time)
        # Channel 3: VAD (broadcasted to frequency)
        
        freq_bins, time_frames = magnitude.shape
        
        ch1 = log_mag
        ch2 = np.tile(log_noise_mag[:, np.newaxis], (1, time_frames))
        ch3 = np.tile(vad_active.astype(np.float32)[np.newaxis, :], (freq_bins, 1))
        
        # Normalize Channel 1 & 2
        global_min = np.min(ch1)
        global_max = np.max(ch1)
        denom = global_max - global_min + 1e-8
        
        ch1_norm = (ch1 - global_min) / denom
        ch2_norm = (ch2 - global_min) / denom  # Use same scale
        
        features = np.stack([ch1_norm, ch2_norm, ch3], axis=-1)
        return features, magnitude, phase, global_min, denom

    def denoise(self, noisy_audio):
        """
        Denoise audio using the advanced mask-based model
        """
        features, orig_magnitude, phase, g_min, g_denom = self.prepare_features(noisy_audio)
        
        freq_bins, time_frames, _ = features.shape
        target_freq = 513
        target_time = 200
        
        # Dimension matching (Pad/Crop)
        padded_features = features.copy()
        
        if freq_bins < target_freq:
            pad_freq = target_freq - freq_bins
            padded_features = np.pad(padded_features, ((0, pad_freq), (0, 0), (0, 0)), mode='constant')
        elif freq_bins > target_freq:
            padded_features = padded_features[:target_freq, :, :]
            
        if time_frames < target_time:
            pad_time = target_time - time_frames
            padded_features = np.pad(padded_features, ((0, 0), (0, pad_time), (0, 0)), mode='constant')
        elif time_frames > target_time:
             # Process in chunks or just crop if it's a demo. Here we crop for simplicity
             # matching the original training paradigm constraints.
             padded_features = padded_features[:, :target_time, :]
             
        # Predict Mask
        model_input = padded_features[np.newaxis, ...]  # (1, 513, 200, 3)
        predicted_mask = self.model.predict(model_input, verbose=0)[0, :, :, 0]
        
        # Revert cropping/padding for mask
        if time_frames < target_time:
            predicted_mask = predicted_mask[:, :time_frames]
        elif time_frames > target_time:
            # We truncated features, so mask is truncated. 
            # We'll just append 1s (no masking) for the missing tail
            padded_mask = np.ones((target_freq, time_frames))
            padded_mask[:, :target_time] = predicted_mask
            predicted_mask = padded_mask
            
        if freq_bins < target_freq:
            predicted_mask = predicted_mask[:freq_bins, :]
            
        # Apply mask to original magnitude
        clean_magnitude = orig_magnitude * predicted_mask
        
        # Spectral flooring to avoid extreme musical noise (noise-fill concept)
        noise_mag, _ = estimate_noise_profile(orig_magnitude)
        noise_floor = noise_mag[:, np.newaxis] * 0.05  # -26dB floor
        clean_magnitude = np.maximum(clean_magnitude, noise_floor)
        
        # ISTFT
        stft = clean_magnitude * np.exp(1j * phase)
        clean_audio = librosa.istft(stft, hop_length=self.hop_length)
        
        # Ensure length match
        if len(clean_audio) > len(noisy_audio):
            clean_audio = clean_audio[:len(noisy_audio)]
        elif len(clean_audio) < len(noisy_audio):
            clean_audio = np.pad(clean_audio, (0, len(noisy_audio) - len(clean_audio)))
            
        return clean_audio
