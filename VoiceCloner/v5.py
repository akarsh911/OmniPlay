from TTS.api import TTS
import os
import json
import soundfile as sf
import numpy as np
import sys

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
SAMPLE_RATE = 24000  # XTTS uses 24 kHz output

# === Generate and Collect Segments ===
final_track = np.zeros(int(total_duration * SAMPLE_RATE), dtype=np.float32)

for i, segment in enumerate(transcription["speech"]):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    speaker_id = segment["speaker_id"]
    emotion = segment.get("emotion", "neutral")
    duration = end - start

    speaker_sample_path = os.path.join(samples_folder, f"speaker{speaker_id}_full.wav")
    output_clip_path = os.path.join(output_folder, f"clip_{i:02d}_{speaker_id}.wav")

    print(f"🎙️ [{i}] Generating: {output_clip_path} | Speaker: {speaker_id} | {duration:.2f}s")

    try:
        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_sample_path,
            file_path=output_clip_path,
            emotion=emotion,
            language="en"  # or "hi"
        )

        # Adjust duration
        audio, sr = sf.read(output_clip_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Sample rate mismatch: expected {SAMPLE_RATE}, got {sr}")

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # convert to mono

        target_samples = int(duration * SAMPLE_RATE)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        # Place in final track
        start_index = int(start * SAMPLE_RATE)
        final_track[start_index:start_index + target_samples] += audio  # add to mix

    except Exception as e:
        print(f"[error] Error processing segment {i}: {e}")

# === Save final mixed audio ===
sf.write(final_output_path, final_track, SAMPLE_RATE)
print(f"\n[success] Final mixed audio saved at: {final_output_path}")
