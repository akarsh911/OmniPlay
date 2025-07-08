#!/usr/bin/env python3
"""
Robust Video Dubbing System - Step 1: Audio Extraction
This version includes fallback methods for compatibility issues.
"""

import os
import subprocess
from pathlib import Path
import sys
# import sys
import platform

if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')


class RobustAudioExtractor:
    def __init__(self, input_video_path, output_dir="temp/step1"):
        self.input_video_path = Path(input_video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.raw_audio_path = self.output_dir / f"{self.input_video_path.stem}_raw_audio.wav"
        self.vocals_path = self.output_dir / f"{self.input_video_path.stem}_vocals.wav"
        self.background_path = self.output_dir / f"{self.input_video_path.stem}_background.wav"
        
        # Check available libraries
        self.has_librosa = self._check_librosa()
        self.has_spleeter = self._check_spleeter()
    
    def _check_librosa(self):
        """Check if librosa is available and working"""
        try:
            import librosa
            import numpy as np
            print(" Librosa is available", flush=True)
            return True
        except Exception as e:
            print(f"Librosa not available: {e}", flush=True)
            return False
    
    def _check_spleeter(self):
        """Check if spleeter is available"""
        try:
            import spleeter
            print("[passed] Spleeter is available", flush=True)
            return True
        except ImportError:
            print("[warning] Spleeter not available", flush=True)
            print(ImportError, flush=True)
            return False
    
    def extract_audio_ffmpeg(self):
        """Extract audio from video using FFmpeg"""
        print(f"🎵 Extracting audio from {self.input_video_path}...", flush=True)
        
        cmd = [
            'ffmpeg', 
            '-i', str(self.input_video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '44100',  # 44.1kHz sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            str(self.raw_audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[passed] Audio extracted successfully: {self.raw_audio_path}", flush=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[error] Error extracting audio: {e}", flush=True)
            print(f"FFmpeg output: {e.stderr}", flush=True)
            return False
        except FileNotFoundError:
            print("[error] FFmpeg not found. Please install FFmpeg and add it to PATH", flush=True)
            return False
    
    def separate_vocals_ffmpeg_simple(self):
        """Simple vocal separation using FFmpeg filters"""
        print(" Using FFmpeg-based vocal separation...", flush=True)
        
        try:
            # Extract center channel (vocals) - simple karaoke effect
            vocals_cmd = [
                'ffmpeg',
                '-i', str(self.raw_audio_path),
                '-af', 'pan=stereo|c0=0.5*c0+-0.5*c1|c1=0.5*c0+-0.5*c1',  # Center channel extraction
                '-ar', '44100',  # Keep sample rate
                '-y',
                str(self.vocals_path)
            ]
            
            # Extract sides (background) - inverse of vocals
            background_cmd = [
                'ffmpeg',
                '-i', str(self.raw_audio_path),
                '-af', 'pan=stereo|c0=c0-0.5*c1|c1=c1-0.5*c0',  # Side channels
                '-ar', '44100',  # Keep sample rate
                '-y',
                str(self.background_path)
            ]
            
            # Run commands
            result1 = subprocess.run(vocals_cmd, check=True, capture_output=True)
            result2 = subprocess.run(background_cmd, check=True, capture_output=True)
            
            # Verify files were created and have content
            if self.vocals_path.exists() and self.background_path.exists():
                vocals_size = self.vocals_path.stat().st_size / (1024*1024)
                background_size = self.background_path.stat().st_size / (1024*1024)
                
                print(f"[passed] Vocals saved to: {self.vocals_path} ({vocals_size:.2f} MB)", flush=True)
                print(f"[passed] Background saved to: {self.background_path} ({background_size:.2f} MB)", flush=True)
                
                if vocals_size > 0 and background_size > 0:
                    return True
                else:
                    print("[warning] Files created but are empty", flush=True)
                    return False
            else:
                print("[warning] Files not created", flush=True)
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"[error] Error in FFmpeg separation: {e}", flush=True)
            print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No error message'}", flush=True)
            return False
    
    def separate_vocals_librosa(self):
        """Advanced vocal separation using librosa"""
        print(" Using Librosa for advanced vocal separation...", flush=True)
        
        try:
            import librosa
            import soundfile as sf
            import numpy as np
            
            # Load audio
            y, sr = librosa.load(str(self.raw_audio_path), sr=None)
            print(f" Loaded audio: {len(y)} samples at {sr}Hz", flush=True)
            
            # Use harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # More advanced separation using spectral analysis
            S_full, phase = librosa.magphase(librosa.stft(y))
            S_filter = librosa.decompose.nn_filter(S_full,
                                                 aggregate=np.median,
                                                 metric='cosine',
                                                 width=int(librosa.time_to_frames(2, sr=sr)))
            S_filter = np.minimum(S_full, S_filter)
            
            margin_i, margin_v = 2, 10
            power = 2
            
            # Create masks for separation
            mask_i = librosa.util.softmask(S_filter,
                                         margin_i * (S_full - S_filter),
                                         power=power)
            
            mask_v = librosa.util.softmask(S_full - S_filter,
                                         margin_v * S_filter,
                                         power=power)
            
            # Apply masks
            S_foreground = mask_v * S_full
            S_background = mask_i * S_full
            
            # Convert back to audio
            vocals = librosa.istft(S_foreground * phase, length=len(y))
            background = librosa.istft(S_background * phase, length=len(y))
              # Save separated audio
            sf.write(str(self.vocals_path), vocals, sr)
            sf.write(str(self.background_path), background, sr)
            
            print(f"[passed] Vocals saved to: {self.vocals_path}", flush=True)
            print(f"[passed] Background saved to: {self.background_path}", flush=True)
            return True
        
        except Exception as e:
            print(f"[error] Error in Librosa separation: {e}", flush=True)
            return False
    
    def separate_vocals_spleeter(self):
        """Professional vocal separation using Spleeter"""
        print(" Using Spleeter for professional vocal separation...", flush=True)
        
        try:
            from spleeter.separator import Separator
            import librosa
            import soundfile as sf
            
            # Initialize separator
            separator = Separator('spleeter:2stems-16kHz')
            
            # Load audio properly for spleeter
            waveform, sr = librosa.load(str(self.raw_audio_path), sr=16000, mono=False)
            print(f" Loaded audio shape: {waveform.shape}, sr: {sr}", flush=True)
            
            # Ensure stereo format for spleeter
            if len(waveform.shape) == 1:
                # Mono to stereo
                waveform = np.stack([waveform, waveform], axis=0)
            elif waveform.shape[0] == 2:
                # Already stereo, keep as is
                pass
            else:
                # Multi-channel, take first two
                waveform = waveform[:2]
            
            print(f" Processing audio shape: {waveform.shape}", flush=True)
            
            # Separate - need to transpose for spleeter
            waveform_t = waveform.T
            prediction = separator.separate(waveform_t)
            
            # Extract results
            vocals = prediction['vocals']
            background = prediction['accompaniment']
            
            print(f" Vocals shape: {vocals.shape}, Background shape: {background.shape}", flush=True)
            
            # Convert back to original sample rate and save
            if vocals.shape[0] > 0:
                # Resample back to 44100 Hz if needed
                if sr != 44100:
                    vocals_44k = librosa.resample(vocals.T, orig_sr=sr, target_sr=44100)
                    background_44k = librosa.resample(background.T, orig_sr=sr, target_sr=44100)
                else:
                    vocals_44k = vocals.T
                    background_44k = background.T
                
                # Save as stereo
                sf.write(str(self.vocals_path), vocals_44k.T, 44100)
                sf.write(str(self.background_path), background_44k.T, 44100)
                
                print(f"[passed] Vocals saved to: {self.vocals_path}", flush=True)
                print(f"[passed] Background saved to: {self.background_path}", flush=True)
                
                # Verify files were created
                if self.vocals_path.exists() and self.background_path.exists():
                    vocals_size = self.vocals_path.stat().st_size / (1024*1024)
                    background_size = self.background_path.stat().st_size / (1024*1024)
                    print(f"   Vocals file size: {vocals_size:.2f} MB", flush=True)
                    print(f"   Background file size: {background_size:.2f} MB", flush=True)
                    
                    if vocals_size > 0 and background_size > 0:
                        return True
                    else:
                        print("[warning] Files created but are empty, trying fallback...", flush=True)
                        return False
                else:
                    print("[warning] Files not created, trying fallback...", flush=True)
                    return False
            else:
                print("[warning] No audio data returned from Spleeter", flush=True)
                return False
            
        except Exception as e:
            print(f"[error] Error in Spleeter separation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def separate_vocals_smart(self):
        """Smart vocal separation using best available method"""
        print("Starting smart vocal separation...", flush=True)
        
        # Try methods in order of quality
        if self.has_spleeter:
            print("[trying] Attempting Spleeter (highest quality)...", flush=True)
            if self.separate_vocals_spleeter():
                return True
        
        if self.has_librosa:
            print("[trying] Attempting Librosa (good quality)...", flush=True)
            if self.separate_vocals_librosa():
                return True
        
        print("[trying] Falling back to FFmpeg (basic quality)...", flush=True)
        return self.separate_vocals_ffmpeg_simple()
    
    def get_audio_info(self):
        """Get information about the extracted audio"""
        if not self.raw_audio_path.exists():
            return None
        
        try:
            # Use FFprobe to get audio info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', str(self.raw_audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                audio_stream = None
                for stream in data['streams']:
                    if stream['codec_type'] == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream:
                    return {
                        'sample_rate': audio_stream.get('sample_rate', 'Unknown'),
                        'duration': float(data['format'].get('duration', 0)),
                        'channels': audio_stream.get('channels', 'Unknown'),
                        'codec': audio_stream.get('codec_name', 'Unknown'),
                        'file_size': int(data['format'].get('size', 0)) / (1024*1024),  # MB
                        'file_path': str(self.raw_audio_path)
                    }
        except Exception as e:
            print(f"Could not get audio info: {e}", flush=True)
        
        return None
    
    def process(self):
        """Run the complete audio extraction process"""
        print(" Starting robust audio extraction process...", flush=True)
        print(f"Input video: {self.input_video_path}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)
        
        # Check if input file exists
        if not self.input_video_path.exists():
            print(f"[error] Input video file not found: {self.input_video_path}", flush=True)
            return False
        
        # Step 1: Extract audio from video
        print("\nStep 1: Extracting audio...", flush=True)
        if not self.extract_audio_ffmpeg():
            return False
        
        # Step 2: Separate vocals and background
        print("\nStep 2: Separating vocals and background...", flush=True)
        if not self.separate_vocals_smart():
            print("[warning] Vocal separation failed, but raw audio is available", flush=True)
            # Continue anyway - we can still proceed with raw audio
        
        # Step 3: Display info
        print("\nStep 3: Audio analysis...", flush=True)
        info = self.get_audio_info()
        if info:
            print("Audio Information:", flush=True)
            for key, value in info.items():
                if key == 'file_size':
                    print(f"  {key}: {value:.2f} MB", flush=True)
                elif key == 'duration':
                    print(f"  {key}: {value:.2f} seconds", flush=True)
                else:
                    print(f"  {key}: {value}", flush=True)
        
        print("\n[passed] Audio extraction completed successfully!", flush=True)
        print(f" Files available in: {self.output_dir}", flush=True)
        
        # List generated files
        for file_path in self.output_dir.glob("*.wav"):
            size = file_path.stat().st_size / (1024*1024)
            print(f"   - {file_path.name} ({size:.2f} MB)", flush=True)
        
        return True

# Usage example
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("[error] Usage: python extract_v0.py <input_video_path>", flush=True)
        print("Example: python extract_v0.py input/input.mp4", flush=True)
        print("Example: python extract_v0.py C:/path/to/your/video.mp4", flush=True)
        sys.exit(1)
    
    # Get input file from command line argument
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"[error] Input file not found: {input_file}", flush=True)
        print("Please ensure your video file exists at the specified path.", flush=True)
        sys.exit(1)
    
    # Initialize and process
    extractor = RobustAudioExtractor(input_file)
    success = extractor.process()
    
    if success:
        print("\n[comlete] Ready for Step 2: Speech Transcription!", flush=True)
    else:
        print("\n[error] Process failed. Please check the errors above.", flush=True)