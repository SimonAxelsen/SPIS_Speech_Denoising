"""
Speech Denoising Comparison: Wiener vs CNN
1. Generate TTS
2. Adjust noise/pitch with live preview
3. Compare Wiener and CNN denoising side-by-side
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
from piper.voice import PiperVoice

from modules.cnn_denoiser import AudioDenoiser, simple_wiener_filter
from modules.advanced_cnn_denoiser import AdvancedAudioDenoiser
from modules.noise_utils import add_noise_at_snr, calculate_snr, generate_noise_by_type
from modules.pitch_shifter import pseudo_cepstrum_pitch_shift


MODEL_DIR = "models"
# MODEL_PATH = os.path.join(MODEL_DIR, "denoiser_trained.h5")
MODEL_PATH = os.path.join(MODEL_DIR, "advanced_denoiser_trained.h5")
PIPER_NAME = "en_US-lessac-low"
PIPER_BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/lessac/low/"
)
SAMPLE_RATE = 22050

os.makedirs(MODEL_DIR, exist_ok=True)


def ensure_piper_model():
    for ext in [".onnx", ".onnx.json"]:
        fname = f"{PIPER_NAME}{ext}"
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(PIPER_BASE_URL + fname, path)


ensure_piper_model()
voice = PiperVoice.load(
    str(Path(MODEL_DIR) / f"{PIPER_NAME}.onnx"),
    str(Path(MODEL_DIR) / f"{PIPER_NAME}.onnx.json"),
)

with open(Path(MODEL_DIR) / f"{PIPER_NAME}.onnx.json", encoding="utf-8") as f:
    VOICE_SR = json.load(f)["audio"]["sample_rate"]

if "advanced" in MODEL_PATH:
    DENOISER = AdvancedAudioDenoiser(model_path=MODEL_PATH, sr=SAMPLE_RATE)
else:
    DENOISER = AudioDenoiser(model_path=MODEL_PATH, sr=SAMPLE_RATE)


def generate_tts(text):
    if not text.strip():
        return None
    audio_chunks = []
    for chunk in voice.synthesize(text):
        pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        audio_chunks.append(pcm)
    if not audio_chunks:
        return None
    audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio


def apply_transforms(clean_audio, pitch_shift, noise_type, snr_db):
    """Apply pitch shift and noise to clean audio"""
    if isinstance(clean_audio, tuple):
        _, audio = clean_audio
    else:
        audio = clean_audio
        
    audio = audio.copy()
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    if pitch_shift != 0:
        audio = pseudo_cepstrum_pitch_shift(audio, SAMPLE_RATE, pitch_shift)
    
    if noise_type.lower() != "none":
        duration = len(audio) / SAMPLE_RATE
        noise = generate_noise_by_type(noise_type, duration, SAMPLE_RATE)
        audio = add_noise_at_snr(audio, noise, snr_db)
    
    return audio


def denoise_wiener(noisy_audio):
    """Apply Wiener denoising only"""
    print(f"[Wiener] Input shape: {noisy_audio.shape}")
    wiener = simple_wiener_filter(noisy_audio, SAMPLE_RATE)
    print(f"[Wiener] Output shape: {wiener.shape}")
    return wiener


def denoise_cnn(noisy_audio):
    """Apply CNN denoising only"""
    print(f"[CNN] Input shape: {noisy_audio.shape}")
    cnn = DENOISER.denoise(noisy_audio)
    print(f"[CNN] Output shape: {cnn.shape}")
    return cnn


def denoise_both(noisy_audio):
    """Apply both Wiener and CNN denoising"""
    wiener = denoise_wiener(noisy_audio)
    cnn = denoise_cnn(noisy_audio)
    return wiener, cnn


def process_pipeline(text, pitch_shift, noise_type, snr_db):
    """Full pipeline: TTS -> Transform -> Denoise both -> Compare"""
    start = time.time()
    
    clean = generate_tts(text)
    if clean is None:
        return None, None, None, None, "No TTS text provided"
    
    if VOICE_SR != SAMPLE_RATE:
        clean = librosa.resample(clean, orig_sr=VOICE_SR, target_sr=SAMPLE_RATE)
    
    noisy = apply_transforms(clean, pitch_shift, noise_type, snr_db)
    wiener_out, cnn_out = denoise_both(noisy)
    
    # Ensure same length for SNR calculation and output
    min_len = min(len(clean), len(wiener_out), len(cnn_out))
    clean_trim = clean[:min_len]
    noisy_trim = noisy[:min_len]
    wiener_trim = wiener_out[:min_len]
    cnn_trim = cnn_out[:min_len]
    
    input_snr = calculate_snr(clean_trim, noisy_trim) if noise_type.lower() != "none" else 100.0
    wiener_snr = calculate_snr(clean_trim, wiener_trim) if noise_type.lower() != "none" else 100.0
    cnn_snr = calculate_snr(clean_trim, cnn_trim) if noise_type.lower() != "none" else 100.0
    
    wiener_gain = wiener_snr - input_snr
    cnn_gain = cnn_snr - input_snr
    
    elapsed = time.time() - start
    
    metrics = (
        f"**Processing time:** {elapsed:.2f}s  \n"
        f"**Input SNR:** {input_snr:.1f} dB  \n"
        f"---  \n"
        f"**Wiener:** {wiener_snr:.1f} dB ({wiener_gain:+.1f} dB)  \n"
        f"**CNN:** {cnn_snr:.1f} dB ({cnn_gain:+.1f} dB)"
    )
    
    return (
        (SAMPLE_RATE, clean_trim),
        (SAMPLE_RATE, noisy_trim),
        (SAMPLE_RATE, wiener_trim),
        (SAMPLE_RATE, cnn_trim),
        metrics
    )


def update_preview(clean_audio, pitch_shift, noise_type, snr_db):
    """Live preview of noise/pitch without denoising"""
    if clean_audio is None:
        return None, "Generate TTS first"
    
    if isinstance(clean_audio, tuple):
        _, audio = clean_audio
    else:
        audio = clean_audio
    
    noisy = apply_transforms(audio, pitch_shift, noise_type, snr_db)
    input_snr = calculate_snr(audio, noisy) if noise_type.lower() != "none" else 100.0
    
    return (
        (SAMPLE_RATE, noisy),
        f"Preview - SNR: {input_snr:.1f} dB"
    )


with gr.Blocks(title="Speech Denoising Comparison") as demo:
    gr.Markdown("# Speech Denoising: Wiener vs CNN")
    gr.Markdown("1. Type text and click **Generate TTS**  2. Adjust noise/pitch  3. Click **Compare** to denoise")

    with gr.Row():
        text_input = gr.Textbox(
            value="The quick brown fox jumps over the lazy dog.",
            label="Text",
            lines=2,
        )
        generate_btn = gr.Button("Generate TTS", variant="primary")

    clean_audio = gr.Audio(label="Clean TTS", type="numpy")
    
    with gr.Row():
        pitch_slider = gr.Slider(-12, 12, 0, step=1, label="Pitch Shift")
        noise_type_drop = gr.Dropdown(["None", "White", "Pink", "Babble"], value="White", label="Noise Type")
        snr_slider = gr.Slider(-5, 25, 10, step=1, label="Noise Level (dB)")

    preview_audio = gr.Audio(label="Noisy Preview", autoplay=True)
    preview_status = gr.Markdown()

    compare_btn = gr.Button("Compare Wiener & CNN", variant="primary")

    with gr.Row():
        wiener_audio = gr.Audio(label="Wiener Filter", type="numpy")
        cnn_audio = gr.Audio(label="CNN Denoiser", type="numpy")

    metrics_out = gr.Markdown()

    def on_generate(text):
        audio = generate_tts(text)
        if audio is None:
            return None, "No text provided"
        if VOICE_SR != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=VOICE_SR, target_sr=SAMPLE_RATE)
        return (SAMPLE_RATE, audio), "TTS generated - adjust sliders below"

    generate_btn.click(on_generate, inputs=[text_input], outputs=[clean_audio, preview_status])

    def on_preview(clean, pitch, noise_type, snr_db):
        return update_preview(clean, pitch, noise_type, snr_db)

    pitch_slider.change(on_preview, [clean_audio, pitch_slider, noise_type_drop, snr_slider], [preview_audio, preview_status])
    noise_type_drop.change(on_preview, [clean_audio, pitch_slider, noise_type_drop, snr_slider], [preview_audio, preview_status])
    snr_slider.change(on_preview, [clean_audio, pitch_slider, noise_type_drop, snr_slider], [preview_audio, preview_status])

    def on_compare(text, pitch, noise_type, snr_db):
        return process_pipeline(text, pitch, noise_type, snr_db)

    compare_btn.click(
        on_compare,
        inputs=[text_input, pitch_slider, noise_type_drop, snr_slider],
        outputs=[clean_audio, preview_audio, wiener_audio, cnn_audio, metrics_out]
    )


if __name__ == "__main__":
    demo.launch(show_error=True)