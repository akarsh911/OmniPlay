import os
import whisper
import torch
from pyannote.audio import Pipeline
import librosa
import numpy as np
from datetime import timedelta
import json
from googletrans import Translator
import soundfile as sf
from pydub import AudioSegment
import tempfile
import requests
import subprocess
import platform

class SimpleAudioTranslationPipeline:
    def __init__(self, whisper_model="base", hf_token=None):
        """
        Initialize the audio translation pipeline with minimal dependencies.
        
        Args:
            whisper_model (str): Whisper model size
            hf_token (str): HuggingFace token for speaker diarization
        """
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Initialize translator
        print("Initializing translator...")
        self.translator = Translator()
        
        # Initialize speaker diarization
        if hf_token:
            try:
                print("Loading speaker diarization pipeline...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                print("Speaker diarization ready!")
            except Exception as e:
                print(f"Speaker diarization failed: {e}")
                self.diarization_pipeline = None
        else:
            print("Warning: No HuggingFace token provided. Speaker diarization will be disabled.")
            self.diarization_pipeline = None
        
       
    
    
    def format_timestamp(self, seconds):
        """Convert seconds to HH:MM:SS.mmm format"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
    
    def detect_language(self, audio_file):
        """Detect the language of the audio"""
        print("Detecting language...")
        result = self.whisper_model.transcribe(audio_file, task="transcribe")
        detected_lang = result.get('language', 'unknown')
        
        # Map Whisper language codes to more readable names
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian'
        }
        
        lang_display = lang_names.get(detected_lang, detected_lang)
        print(f"Detected language: {lang_display} ({detected_lang})")
        return detected_lang
    
    def translate_text(self, text, source_lang='auto', target_lang='en'):
        """Translate text using Google Translate"""
        if not text.strip():
            return text
        
        try:
            # Skip translation if already in target language
            if source_lang == target_lang:
                return text
            
            translated = self.translator.translate(text, src=source_lang, dest=target_lang)
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def calculate_speech_rate(self, text, duration):
        """Calculate words per minute for speech rate matching"""
        word_count = len(text.split())
        if duration > 0:
            wpm = (word_count / duration) * 60
            return max(50, min(300, wpm))  # Clamp between 50-300 WPM
        return 150  # Default WPM  
    
    def perform_speaker_diarization(self, audio_file):
        """Perform speaker diarization"""
        if not self.diarization_pipeline:
            return None
        
        print("Performing speaker diarization...")
        try:
            diarization = self.diarization_pipeline(audio_file)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            return speaker_segments
        except Exception as e:
            print(f"Speaker diarization error: {e}")
            return None
    
    def transcribe_segment(self, audio_file, start_time, end_time):
        """Transcribe a specific segment of audio"""
        try:
            audio, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=end_time-start_time)
            result = self.whisper_model.transcribe(audio, task="transcribe")
            return result["text"].strip()
        except Exception as e:
            print(f"Warning: Could not transcribe segment {start_time}-{end_time}: {e}")
            return ""
    
    def transcribe_with_speakers(self, audio_file):
        """Transcribe audio with speaker identification"""
        print(f"Processing audio file: {audio_file}")
        
        # Full transcription
        try:
            full_result = self.whisper_model.transcribe(audio_file, word_timestamps=True)
        except TypeError:
            full_result = self.whisper_model.transcribe(audio_file)
        
        results = {
            'file': audio_file,
            'duration': 0,
            'language': full_result.get('language', 'unknown'),
            'full_text': full_result['text'],
            'segments': [],
            'speakers': [],
            'speaker_count': 0
        }
        
        # Calculate duration
        if full_result.get('segments'):
            results['duration'] = full_result['segments'][-1].get('end', 0)
        else:
            try:
                audio_data, sr = librosa.load(audio_file, sr=None)
                results['duration'] = len(audio_data) / sr
            except:
                results['duration'] = 0
        
        # Speaker diarization
        if self.diarization_pipeline:
            speaker_segments = self.perform_speaker_diarization(audio_file)
            
            if speaker_segments:
                unique_speakers = list(set([seg['speaker'] for seg in speaker_segments]))
                results['speakers'] = unique_speakers
                results['speaker_count'] = len(unique_speakers)
                
                print(f"Detected {len(unique_speakers)} speakers: {', '.join(unique_speakers)}")
                
                for segment in speaker_segments:
                    text = self.transcribe_segment(audio_file, segment['start'], segment['end'])
                    
                    if text:
                        results['segments'].append({
                            'start_time': segment['start'],
                            'end_time': segment['end'],
                            'start_timestamp': self.format_timestamp(segment['start']),
                            'end_timestamp': self.format_timestamp(segment['end']),
                            'speaker': segment['speaker'],
                            'text': text
                        })
        else:
            # Fallback without speaker identification
            print("Using Whisper segmentation (no speaker identification)...")
            if full_result.get('segments'):
                for segment in full_result['segments']:
                    results['segments'].append({
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'start_timestamp': self.format_timestamp(segment['start']),
                        'end_timestamp': self.format_timestamp(segment['end']),
                        'speaker': 'Speaker_01',
                        'text': segment['text'].strip()
                    })
                
                results['speaker_count'] = 1
                results['speakers'] = ['Speaker_01']
        
        return results
    
    def process_complete_pipeline(self, audio_file, output_dir="translated_audio", target_language="en"):
        """Complete pipeline: transcribe, translate, and generate TTS audio"""
        print(f"Starting complete translation pipeline for: {audio_file}")
        print(f"Available TTS options: {[k for k, v in self.system_tts.items() if v]}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Detect language
        source_language = self.detect_language(audio_file)
        
        # Step 2: Transcribe with speaker identification
        print("Transcribing audio with speaker identification...")
        transcription_results = self.transcribe_with_speakers(audio_file)
        
        # Step 3: Translate transcriptions
        print("Translating transcriptions...")
        translated_results = transcription_results.copy()
        
        for segment in translated_results['segments']:
            original_text = segment['text']
            translated_text = self.translate_text(
                original_text, 
                source_lang=source_language, 
                target_lang=target_language
            )
            segment['original_text'] = original_text
            segment['translated_text'] = translated_text
            segment['target_duration'] = segment['end_time'] - segment['start_time']
        
        # Step 4: Generate translated audio
        print("Generating translated audio...")
       
        
      
        results_file = os.path.join(output_dir, "translation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': audio_file,
                'source_language': source_language,
                'target_language': target_language,
                'transcription_results': translated_results,
            }, f, indent=2, ensure_ascii=False)
        
        return {
            'results_file': results_file,
            'total_speakers': translated_results['speaker_count'],
            'source_language': source_language
        }
    
   
def main():
    """Example usage with simplified pipeline"""
    
    # Configuration
    AUDIO_FILE = "extracted_audio/input_vocals.wav"  # Updated to match your file
    WHISPER_MODEL = "tiny"  # Options: tiny, base, small, medium, large
    HF_TOKEN = "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"  # Add your HuggingFace token here for speaker diarization
    OUTPUT_DIR = "translated_output_v3"
    TARGET_LANGUAGE = "en"  # English
      
    # Initialize pipeline
    pipeline = SimpleAudioTranslationPipeline(
        whisper_model=WHISPER_MODEL,
        hf_token=HF_TOKEN
    )
    
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file '{AUDIO_FILE}' not found!")
        print("Please update the AUDIO_FILE path in the script.")
        return
    
    try:
        # Run complete pipeline
        results = pipeline.process_complete_pipeline(
            audio_file=AUDIO_FILE,
            output_dir=OUTPUT_DIR,
            target_language=TARGET_LANGUAGE
        )
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Results file: {results['results_file']}")
        print(f"🎵 Final translated audio: {results['final_audio']}")
        print(f"📊 Generated segments: {results['generated_segments']}")
        print(f"👥 Total speakers: {results['total_speakers']}")
        print(f"🌐 Source language: {results['source_language']}")
        
        if results['final_audio']:
            print(f"\n✅ Success! Check the '{OUTPUT_DIR}' folder for:")
            print(f"   • Individual translated segments")
            print(f"   • Complete translated audio file")
            print(f"   • Detailed JSON results")
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Install missing TTS: pip install edge-tts pyttsx3")
        print("2. Update protobuf: pip install 'protobuf<4.0.0'")
        print("3. Get HuggingFace token for speaker diarization")
        print("4. Ensure audio file exists and is readable")

if __name__ == "__main__":
    main()

# Installation instructions:
"""
Fixed installation to avoid dependency conflicts:

pip install openai-whisper pyannote.audio librosa torch googletrans==4.0.0rc1 pydub soundfile 'protobuf<4.0.0'

For TTS options (install one or more):
pip install edge-tts          # Best quality, requires internet
pip install pyttsx3           # Offline, basic quality

Additional notes:
- Downgraded protobuf to avoid TensorFlow conflicts
- Removed TTS library dependency (was causing conflicts)
- Added multiple TTS fallback options
- Edge TTS provides the best voice quality
- pyttsx3 works offline but with basic voices
"""