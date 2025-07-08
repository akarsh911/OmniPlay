from TTS.api import TTS
import os
import sys

# === Check command line arguments ===
if len(sys.argv) < 3:
    print("[error] Usage: python v3.py <reference_audio_path> <text_to_synthesize>")
    print("Example: python v3.py samples/lovee_clean.wav \"Hello, this is a test.\"")
    print("Example: python v3.py \"C:/path/to/reference.wav\" \"Your text here\"")
    sys.exit(1)

# === Config ===
reference_audio_path = sys.argv[1]
text = sys.argv[2]
output_audio_path = "./output/cloned_speech.wav"

# === Check if input file exists ===
if not os.path.exists(reference_audio_path):
    print(f"[error] Reference audio file not found: {reference_audio_path}")
    print("Please ensure your reference audio file exists at the specified path.")
    sys.exit(1)

# === Load TTS model ===
# Use a multilingual multi-speaker model (supports speaker_wav cloning)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

print("[loading] Loading TTS model...")
tts = TTS(model_name=model_name, progress_bar=True, gpu=True)  # set gpu=True if you have one

# === Ensure folders exist ===
os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
# embed = tts.get_speaker_embedding("voice.wav")

# === Generate TTS using the reference speaker ===
print("🎙️ Generating cloned voice...")
tts.tts_to_file(
    text=text,
    speaker_wav=reference_audio_path,
    file_path=output_audio_path,
    emotion="excited",  # Optional: specify emotion if supported
    language="hi"
)

print(f"[success] Speech saved at {output_audio_path}")
print(f"🎙️ Generated speech using reference: {reference_audio_path}")
print(f"📝 Text synthesized: {text}")
