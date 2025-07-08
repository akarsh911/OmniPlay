from TTS.api import TTS
import os
import json
import soundfile as sf
import numpy as np
import sys
import librosa
from scipy.signal import butter, filtfilt
import warnings
import platform
warnings.filterwarnings("ignore")
# import sys

if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')

# === Config ===
transcription_json_path = "./temp/step2/output.json"
samples_folder = "./temp/step2/samples"
output_folder = "./temp/step3/output_clips"
final_output_path = "./temp/step3/final_mix.wav"

# === Validate JSON file ===
if not os.path.exists(transcription_json_path):
    print(f"[error] Transcription file not found: {transcription_json_path}")
    sys.exit(1)

with open(transcription_json_path, "r", encoding="utf-8") as f:
    transcription = json.load(f)

h, m, s = map(int, transcription["total_length"].split(":"))
total_duration = h * 3600 + m * 60 + s

# === Load TTS ===
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("[loading] Loading TTS model...")
tts = TTS(model_name=model_name, progress_bar=True, gpu=True)

# === Ensure folders ===
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

SAMPLE_RATE = 22050
TARGET_SAMPLE_RATE = 24000

def enhance_audio(audio, sr):
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    threshold = 0.3
    ratio = 4.0
    compressed = np.where(np.abs(audio) > threshold, np.sign(audio)*(threshold+(np.abs(audio)-threshold)/ratio), audio)
    gated = np.where(np.abs(compressed) < 0.01, compressed * 0.1, compressed)
    nyq = sr / 2
    b, a = butter(2, 80/nyq, btype='high'); filtered = filtfilt(b, a, gated)
    b, a = butter(2, 8000/nyq, btype='low'); filtered = filtfilt(b, a, filtered)
    return filtered * 0.8

def resize_audio(audio, target_len, sr):
    cur_len = len(audio)
    if cur_len == target_len: return audio
    if cur_len > target_len: return audio[:target_len]
    pad = target_len - cur_len
    fade = min(1000, cur_len // 4)
    if fade > 0:
        audio[-fade:] *= np.linspace(1, 0, fade)
    return np.pad(audio, (0, pad), mode='constant')

def process_sample(path, dur=3.0):
    if not os.path.exists(path): return None
    audio, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio) < 1.0 * sr: return None
    return audio[:int(dur * sr)]

final_track = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)
processed_samples = {}

for seg in transcription["speech"]:
    sid = seg["speaker_id"]
    if sid not in processed_samples:
        p = os.path.join(samples_folder, f"speaker{sid}_full.wav")
        audio = process_sample(p)
        if audio is not None:
            temp = os.path.join(samples_folder, f"speaker{sid}_processed.wav")
            sf.write(temp, audio, TARGET_SAMPLE_RATE)
            processed_samples[sid] = temp
        else:
            processed_samples[sid] = p

print(f" Processing {len(transcription['speech'])} segments...")
for i, seg in enumerate(transcription["speech"]):
    start, end = seg["start"], seg["end"]
    text = seg["translation_llm"].strip()
    sid = seg["speaker_id"]
    emo = seg.get("emotion", "neutral")
    dur = end - start

    if not text:
        print(f"[warning] [{i}] Empty segment.")
        continue

    spath = processed_samples.get(sid)
    outpath = os.path.join(output_folder, f"clip_{i:02d}_{sid}.wav")

    try:
        tts.tts_to_file(text=text, speaker_wav=spath, file_path=outpath, emotion=emo, language="en", speed=1.0)
        audio, sr = sf.read(outpath)
        if sr != TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = enhance_audio(audio, sr)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = resize_audio(audio, int(dur * SAMPLE_RATE), SAMPLE_RATE)
        s_idx = int(start * SAMPLE_RATE)
        e_idx = s_idx + len(audio)
        final_track[s_idx:e_idx] += audio[:min(len(audio), len(final_track) - s_idx)]
        sf.write(outpath, audio, SAMPLE_RATE)
    except Exception as e:
        print(f"[error] Segment {i} failed: {e}")

# Final mix processing
print("\n[loading] Enhancing final mix...")
final_track = final_track / np.max(np.abs(final_track) + 1e-8) * 0.95
final_track = enhance_audio(final_track, SAMPLE_RATE)
sf.write(final_output_path, final_track, SAMPLE_RATE)
print(f"[success] Final mix saved at: {final_output_path}")
