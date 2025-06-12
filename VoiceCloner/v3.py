from TTS.api import TTS
import os

# === Config ===
reference_audio_path = "./samples/lovee_clean.wav"
output_audio_path = "./output/cloned_speech.wav"
text = "मेरा नाम टीया है और मैं कक्षा 9 में सीबीएससी में पढ़ती हूँ। मुझे दूध पीना पसंद नहीं है। "

# === Load TTS model ===
# Use a multilingual multi-speaker model (supports speaker_wav cloning)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

print("🔄 Loading TTS model...")
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

print(f"✅ Speech saved at {output_audio_path}")
