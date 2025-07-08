from TTS.api import TTS
import os
import json
import soundfile as sf
import numpy as np
import sys
import librosa
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

# === Check command line arguments ===
if len(sys.argv) < 2:
    print("[error] Usage: python v5.py <transcription_json_path>")
    print("Example: python v5.py input.json")
    print("Example: python v5.py C:/path/to/transcription.json")
    sys.exit(1)

# === Config ===
transcription_json_path = sys.argv[1]
samples_folder = "./samples"
output_folder = "./output_clips"
final_output_path = "./output/final_mix.wav"

# === Check if input file exists ===
if not os.path.exists(transcription_json_path):
    print(f"[error] Transcription file not found: {transcription_json_path}")
    print("Please ensure your JSON file exists at the specified path.")
    sys.exit(1)

# === Load JSON with diarized segments ===
with open(transcription_json_path, "r", encoding="utf-8") as f:
    transcription = json.load(f)

total_length_str = transcription["total_length"]  # Format: "0:00:31"
h, m, s = map(int, total_length_str.split(":"))
total_duration = h * 3600 + m * 60 + s

# === Load TTS model ===
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("[loading] Loading TTS model...")
tts = TTS(model_name=model_name, progress_bar=True, gpu=True)

# === Ensure folders exist ===
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

# === Settings ===
SAMPLE_RATE = 22050  # Standard rate for better compatibility
TARGET_SAMPLE_RATE = 24000  # XTTS native rate

def enhance_audio_naturalness(audio, sample_rate):
    """Apply audio enhancements to make speech sound more natural"""
    
    # 1. Normalize audio to prevent clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # 2. Apply subtle compression to even out volume
    threshold = 0.3
    ratio = 4.0
    compressed = np.where(np.abs(audio) > threshold, 
                         np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
                         audio)
    
    # 3. Add subtle noise gate to reduce background noise
    gate_threshold = 0.01
    gated = np.where(np.abs(compressed) < gate_threshold, 
                     compressed * 0.1, 
                     compressed)
    
    # 4. Apply gentle high-pass filter to remove low-frequency artifacts
    nyquist = sample_rate / 2
    high_cutoff = 80 / nyquist
    b, a = butter(2, high_cutoff, btype='high')
    filtered = filtfilt(b, a, gated)
    
    # 5. Apply gentle low-pass filter to smooth harsh frequencies
    low_cutoff = 8000 / nyquist
    b, a = butter(2, low_cutoff, btype='low')
    filtered = filtfilt(b, a, filtered)
    
    # 6. Final normalization
    filtered = filtered * 0.8  # Leave some headroom
    
    return filtered

def safe_audio_resize(audio, target_samples, sample_rate):
    """Safely resize audio to target length with proper handling"""
    current_samples = len(audio)
    
    if current_samples == target_samples:
        return audio
    elif current_samples > target_samples:
        # Trim from the end
        return audio[:target_samples]
    else:
        # Pad with silence, but add a small fade-out to avoid clicks
        padding = target_samples - current_samples
        padded = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        # Add fade-out to original audio end to avoid clicks
        fade_samples = min(1000, current_samples // 4)  # 0.05s fade at 22kHz
        if fade_samples > 0:
            fade_curve = np.linspace(1, 0, fade_samples)
            padded[current_samples - fade_samples:current_samples] *= fade_curve
        
        return padded

def process_speaker_sample(speaker_sample_path, target_duration=3.0):
    """Process and optimize speaker sample for better voice cloning"""
    if not os.path.exists(speaker_sample_path):
        print(f"[warning]  Speaker sample not found: {speaker_sample_path}")
        return None
    
    # Load and preprocess speaker sample
    audio, sr = librosa.load(speaker_sample_path, sr=TARGET_SAMPLE_RATE)
    
    # Trim silence from beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Ensure minimum duration for better cloning
    min_duration = 1.0  # seconds
    if len(audio) < min_duration * sr:
        print(f"[warning]  Speaker sample too short, consider using a longer sample")
    
    # Limit duration to avoid memory issues
    max_samples = int(target_duration * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    return audio

# === Generate and Collect Segments ===
final_track = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)

# Process speaker samples first
processed_speaker_samples = {}
for segment in transcription["speech"]:
    speaker_id = segment["speaker_id"]
    if speaker_id not in processed_speaker_samples:
        speaker_sample_path = os.path.join(samples_folder, f"speaker{speaker_id}_full.wav")
        processed_sample = process_speaker_sample(speaker_sample_path)
        if processed_sample is not None:
            # Save processed sample temporarily
            temp_path = os.path.join(samples_folder, f"speaker{speaker_id}_processed.wav")
            sf.write(temp_path, processed_sample, TARGET_SAMPLE_RATE)
            processed_speaker_samples[speaker_id] = temp_path
        else:
            processed_speaker_samples[speaker_id] = speaker_sample_path

print(f"📊 Processing {len(transcription['speech'])} segments...")

for i, segment in enumerate(transcription["speech"]):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()
    speaker_id = segment["speaker_id"]
    emotion = segment.get("emotion", "neutral")
    duration = end - start

    # Skip empty text
    if not text:
        print(f"[warning]  [{i}] Skipping empty text segment")
        continue

    speaker_sample_path = processed_speaker_samples.get(speaker_id)
    output_clip_path = os.path.join(output_folder, f"clip_{i:02d}_{speaker_id}.wav")

    print(f"🎙️ [{i}] Generating: {output_clip_path} | Speaker: {speaker_id} | {duration:.2f}s")
    print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")

    try:
        # Generate speech with improved settings
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_sample_path,
            file_path=output_clip_path,
            emotion=emotion,
            language="en",  # Change to "hi" for Hindi
            # Additional parameters for better quality
            speed=1.0,
            # split_sentences=True  # Uncomment if available in your TTS version
        )

        # Load and process generated audio
        audio, sr = sf.read(output_clip_path)
        
        # Ensure correct sample rate
        if sr != TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Enhance audio naturalness
        audio = enhance_audio_naturalness(audio, sr)

        # Resample to final sample rate if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Resize audio to match segment duration
        target_samples = int(duration * SAMPLE_RATE)
        audio = safe_audio_resize(audio, target_samples, SAMPLE_RATE)

        # Safely place in final track with bounds checking
        start_index = int(start * SAMPLE_RATE)
        end_index = start_index + target_samples
        
        # Ensure we don't go beyond the final track bounds
        if end_index > len(final_track):
            print(f"[warning]  [{i}] Segment extends beyond total duration, truncating")
            end_index = len(final_track)
            target_samples = end_index - start_index
            audio = audio[:target_samples]

        # Add to final track with proper bounds checking
        if start_index < len(final_track) and target_samples > 0:
            final_track[start_index:end_index] += audio[:target_samples]

        # Save individual processed clip
        sf.write(output_clip_path, audio, SAMPLE_RATE)

    except Exception as e:
        print(f"[error] Error processing segment {i}: {e}")
        print(f"   Segment details: start={start}, end={end}, duration={duration}")

# === Apply final audio processing ===
print("[loading] Applying final audio processing...")

# Normalize final track
max_val = np.max(np.abs(final_track))
if max_val > 0:
    final_track = final_track / max_val * 0.95  # Prevent clipping

# Apply final enhancement
final_track = enhance_audio_naturalness(final_track, SAMPLE_RATE)

# === Save final mixed audio ===
sf.write(final_output_path, final_track, SAMPLE_RATE)
print(f"\n[success] Final mixed audio saved at: {final_output_path}")
print(f"📊 Total duration: {total_duration}s, Sample rate: {SAMPLE_RATE}Hz")

# Clean up temporary processed samples
for speaker_id, temp_path in processed_speaker_samples.items():
    if temp_path.endswith('_processed.wav') and os.path.exists(temp_path):
        os.remove(temp_path)