from TTS.api import TTS
import os
import json
import soundfile as sf
import numpy as np
import sys

# === Check command line arguments ===
if len(sys.argv) < 2:
    print("[error] Usage: python v4.py <transcription_json_path>")
    print("Example: python v4.py input.json")
    print("Example: python v4.py C:/path/to/transcription.json")
    sys.exit(1)

# === Paths & Config ===
transcription_json_path = sys.argv[1]  # Contains diarized speech segments
samples_folder = "./samples"                      # Folder with speaker audio samples
output_folder = "./output_clips"                  # Folder for individual audio clips

# === Check if input file exists ===
if not os.path.exists(transcription_json_path):
    print(f"[error] Transcription file not found: {transcription_json_path}")
    print("Please ensure your JSON file exists at the specified path.")
    sys.exit(1)

# === Load JSON with diarized segments ===
with open(transcription_json_path, "r", encoding="utf-8") as f:
    transcription = json.load(f)

# === Load TTS model ===
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("[loading] Loading TTS model...")
tts = TTS(model_name=model_name, progress_bar=True, gpu=True)

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Generate speech for each segment ===
for i, segment in enumerate(transcription["speech"]):
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    speaker_id = segment["speaker_id"]
    emotion = segment.get("emotion", "neutral")

    duration = end - start
    speaker_sample_path = os.path.join(samples_folder, f"speaker{speaker_id}_full.wav")

    output_clip_path = os.path.join(output_folder, f"clip_{i:02d}_{speaker_id}.wav")
    print(f"🎙️ Generating: {output_clip_path} | Speaker: {speaker_id} | Duration: {duration:.2f}s")

    try:
        # Generate TTS output
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_sample_path,
            file_path=output_clip_path,
            emotion=emotion,
            language="en"  # change to "hi" if your segments are in Hindi
        )

        # === Trim or pad to match exact duration ===
        audio, sr = sf.read(output_clip_path)
        actual_duration = len(audio) / sr
        target_samples = int(duration * sr)

        if len(audio.shape) > 1:
            # Convert to mono if stereo
            audio = np.mean(audio, axis=1)

        if actual_duration > duration:
            audio = audio[:target_samples]
        elif actual_duration < duration:
            pad_length = target_samples - len(audio)
            audio = np.pad(audio, (0, pad_length), mode='constant')

        # Save the corrected-duration audio
        sf.write(output_clip_path, audio, sr)
        print(f"[success] Saved and adjusted: {output_clip_path}")

    except Exception as e:
        print(f"[error] Error processing segment {i}: {e}")

print("\n[comlete] All segments processed.")
