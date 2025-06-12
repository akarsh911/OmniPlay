import os
import json
import torch
from pydub import AudioSegment
from datetime import timedelta
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ----- Configuration -----
AUDIO_FILE = "input.wav"
OUTPUT_JSON = "output.json"
SAMPLES_DIR = "samples"
WHISPER_MODEL_SIZE = "medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------

def debug(msg):
    print(f"[DEBUG] {msg}")

# STEP 1: TRANSCRIPTION (Whisper)
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

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

# STEP 2: SPEAKER DIARIZATION
def diarize(audio_path):
    debug("Loading Pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

    debug("Performing speaker diarization...")
    diarization = pipeline(audio_path)
    
    # Results are segments labeled with speaker IDs
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    debug(f"Diarization complete. Found {len(set([s['speaker'] for s in speaker_segments]))} speakers.")
    return speaker_segments

# STEP 3: ALIGN TRANSCRIPTION TO SPEAKERS
def align(transcript, diarization):
    debug("Aligning transcribed segments to speaker IDs...")

    aligned = []
    for seg in transcript:
        start = seg["start"]
        speaker_id = "unknown"

        for dia in diarization:
            if dia["start"] <= start <= dia["end"]:
                speaker_id = dia["speaker"]
                break

        aligned.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": speaker_id,
            "text": seg["text"],
            "emotion": "neutral"  # Placeholder for future emotion detection
        })

    debug("Alignment complete.")
    return aligned

# STEP 4: SAVE SPEAKER SAMPLES
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
            sample = audio[start_ms : min(end_ms+1000, start_ms + 4000)]
            sample.export(f"{output_dir}/speaker{speaker}_sample.wav", format="wav")
            saved[speaker] = True
            debug(f"Saved sample for speaker {speaker}")

# STEP 5: FORMAT FINAL OUTPUT
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

# MAIN FUNCTION
def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"[ERROR] Audio file '{AUDIO_FILE}' not found.")
        return

    debug("Starting transcription pipeline...")

    # Step 1: Transcribe
    transcript, total_duration, detected_lang = transcribe(AUDIO_FILE)

    # Step 2: Translate original to English
    transcript = translate_segments(transcript, detected_lang)

    # Step 3: Diarization
    diarization = diarize(AUDIO_FILE)

    # Step 4: Align
    aligned_segments = align(transcript, diarization)

    # Step 5: Samples
    save_samples(AUDIO_FILE, diarization, SAMPLES_DIR)

    # Step 6: Output JSON
    write_output(aligned_segments, total_duration, OUTPUT_JSON)

    debug("✅ All steps complete!")

if __name__ == "__main__":
    main()
