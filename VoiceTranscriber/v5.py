import os
import json
import torch
import sys
from pydub import AudioSegment
from datetime import timedelta
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
# import sys
import platform

if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')

# === Check command line arguments ===
if len(sys.argv) < 2:
    print("❌ Usage: python v3.py <input_audio_path> [output_json_path]", flush=True)
    sys.exit(1)

# ----- Configuration -----
AUDIO_FILE = sys.argv[1]
OUTPUT_JSON = sys.argv[2] if len(sys.argv) > 2 else "step2/output.json"
SAMPLES_DIR = "step2/samples"
WHISPER_MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(AUDIO_FILE):
    print(f"❌ Audio file not found: {AUDIO_FILE}", flush=True)
    sys.exit(1)

# -------------------------
def debug(msg):
    print(f"[DEBUG] {msg}", flush=True)

# STEP 1: TRANSCRIPTION (Whisper)
def transcribe(audio_path):
    debug("Loading Whisper model...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE)

    debug("Transcribing and detecting language...")
    segments, info = model.transcribe(audio_path, beam_size=5, language=None)
    language = info.language
    debug(f"Detected language: {language}")

    transcript = []
    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "original_text": segment.text.strip()
        })

    return transcript, info.duration, language

# STEP 2: TRANSLATE

def translate_segments(segments, source_lang):
    debug("Translating original_text to English...")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(DEVICE)
    tokenizer.src_lang = source_lang

    for seg in segments:
        input_ids = tokenizer(seg["original_text"], return_tensors="pt").input_ids.to(DEVICE)
        generated_tokens = model.generate(input_ids, forced_bos_token_id=tokenizer.get_lang_id("en"))
        seg["text"] = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    debug("Translation complete.")
    return segments

# STEP 3: TRANSLATE WITH LLM
import openai

def translate_with_llm(segments):
   debug("Translating with GPT for natural dubbing...")
   for seg in segments:
        prompt = f"Translate this into natural, expressive English suitable for dubbing:\n\n\"{seg['original_text']}\""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            seg["translation_llm"] = response.choices[0].message["content"].strip()
        except Exception as e:
            debug(f"OpenAI translation failed: {e}")
            seg["translation_llm"] = ""
   return segments

# STEP 4: DIARIZATION

def diarize(audio_path):
    debug("Loading Pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    debug("Performing speaker diarization...")
    diarization = pipeline(audio_path)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    debug(f"Diarization complete. Found {len(set([s['speaker'] for s in speaker_segments]))} speakers.")
    return speaker_segments

# STEP 5: ALIGN

def align(transcript, diarization):
    debug("Aligning transcribed segments to speaker IDs...")
    aligned = []
    for seg in transcript:
        seg_start, seg_end = seg["start"], seg["end"]
        best_match = None
        max_overlap = 0

        for dia in diarization:
            dia_start, dia_end = dia["start"], dia["end"]
            overlap = max(0, min(seg_end, dia_end) - max(seg_start, dia_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = dia["speaker"]

        aligned.append({
            "start": seg_start,
            "end": seg_end,
            "speaker_id": best_match if best_match else "unknown",
            "original_text": seg["original_text"],
            "text": seg.get("text", ""),
            "translation_llm": seg.get("translation_llm", ""),
            "emotion": "neutral"  # placeholder, will be updated
        })

    debug("Alignment complete.")
    return aligned

# STEP 6: EMOTION DETECTION

def detect_emotion(audio_path, segments):
    debug("Loading emotion detection model...")
    emotion_model = pipeline("audio-classification", model="superb/hubert-large-superb-er", device=0 if DEVICE == "cuda" else -1)
    audio = AudioSegment.from_file(audio_path)

    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]

        temp_path = "temp_segment.wav"
        chunk.export(temp_path, format="wav")

        try:
            result = emotion_model(temp_path)
            seg["emotion"] = result[0]["label"] if result else "neutral"
        except Exception as e:
            debug(f"Emotion detection failed: {e}")
            seg["emotion"] = "neutral"

        os.remove(temp_path)

    debug("Emotion detection complete.")
    return segments

# STEP 7: SAVE SPEAKER SAMPLES

def save_samples(audio_path, diarization, output_dir="samples"):
    debug(f"Extracting speaker samples to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    saved = {}

    for seg in diarization:
        speaker = seg["speaker"]
        if speaker not in saved:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            sample = audio[start_ms : min(end_ms + 1000, start_ms + 4000)]
            sample.export(f"{output_dir}/speaker{speaker}_sample.wav", format="wav")
            saved[speaker] = True
            debug(f"Saved sample for speaker {speaker}")

# STEP 8: OUTPUT JSON

def write_output(transcription, total_duration, output_path):
    debug("Writing output JSON...")
    speakers = set([s["speaker_id"] for s in transcription])
    output = {
        "total_length": str(timedelta(seconds=int(total_duration))),
        "total_speakers": len(speakers),
        "speech": transcription
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    debug(f"Output written to {output_path}")

# MAIN

def main():
    debug("Starting transcription pipeline...")
    transcript, total_duration, detected_lang = transcribe(AUDIO_FILE)
    transcript = translate_segments(transcript, detected_lang)
    transcript = translate_with_llm(transcript)
    diarization = diarize(AUDIO_FILE)
    aligned_segments = align(transcript, diarization)
    aligned_segments = detect_emotion(AUDIO_FILE, aligned_segments)
    save_samples(AUDIO_FILE, diarization, SAMPLES_DIR)
    write_output(aligned_segments, total_duration, OUTPUT_JSON)
    debug("\u2705 All steps complete!")

if __name__ == "__main__":
    main()
