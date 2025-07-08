import json
import pysrt
import os
import sys
import numpy as np
import platform

if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')

def import_moviepy():
    """Import MoviePy with fallback methods for different versions"""
    print(" Attempting to import MoviePy...", flush=True)

    try:
        import moviepy
        print(f" MoviePy version: {moviepy.__version__}", flush=True)

        # Use main editor module to import all
        from moviepy.editor import VideoFileClip, AudioFileClip, AudioClip, concatenate_audioclips, CompositeAudioClip
        print(" Imported from moviepy.editor", flush=True)
        return VideoFileClip, AudioFileClip, AudioClip, concatenate_audioclips, CompositeAudioClip

    except ImportError as e:
        print(f" MoviePy import failed: {e}", flush=True)
        return None, None, None, None, None

    
# Import MoviePy classes
VideoFileClip, AudioFileClip, AudioClip, concatenate_audioclips, CompositeAudioClip = import_moviepy()

if not VideoFileClip or not AudioFileClip:
    print(" Could not import required MoviePy classes", flush=True)
    sys.exit(1)

# Check command line arguments
if len(sys.argv) < 2:
    print(" Usage: python mixer.py <input_video_path>", flush=True)
    print("Example: python mixer.py input.mp4", flush=True)
    print("Example: python mixer.py \"C:/path/to/your/video.mp4\"", flush=True)
    print("", flush=True)
    print(" Note: Creates video with multiple audio tracks (original + dubbed)", flush=True)
    sys.exit(1)

# Paths
json_path = "temp/step2/output.json"
video_path = sys.argv[1]  # Get input video from command line argument
tts_audio_path = "temp/step3/final_mix.wav"
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create video with multiple audio tracks
output_video_path = f"C:/Dubbed/video_{timestamp}/video_multi_audio.mp4"
print(" Creating video with multiple audio tracks (original + dubbed)", flush=True)

srt_path = f"C:/Dubbed/video_{timestamp}/subtitles.srt"

# Check if input video exists
if not os.path.exists(video_path):
    print(f" Input video file not found: {video_path}", flush=True)
    print("Please ensure your video file exists at the specified path.", flush=True)
    sys.exit(1)

from moviepy.audio.AudioClip import AudioArrayClip

def create_silence(duration, fps=44100):
    """Create silence as a proper stereo AudioClip"""
    total_samples = int(duration * fps)
    silence_array = np.zeros((total_samples, 2), dtype=np.float32)  # stereo: 2 channels
    return AudioArrayClip(silence_array, fps=fps)

def mix_tts_with_background():
    """Mix TTS audio with background audio"""
    try:
        background_audio_path = "temp/step1/input_background.wav"
        
        print(" Loading TTS audio...", flush=True)
        if not os.path.exists(tts_audio_path):
            print(f" TTS audio file not found: {tts_audio_path}", flush=True)
            return None
        
        tts_audio = AudioFileClip(tts_audio_path)
        print(f"   TTS audio duration: {tts_audio.duration:.2f} seconds", flush=True)
        
        print(" Loading background audio...", flush=True)
        if not os.path.exists(background_audio_path):
            print(f" Background audio file not found: {background_audio_path}", flush=True)
            print(" Using only TTS audio (no background)", flush=True)
            return tts_audio
        
        background_audio = AudioFileClip(background_audio_path)
        print(f"   Background audio duration: {background_audio.duration:.2f} seconds", flush=True)
        
        # Adjust TTS audio duration to match background audio duration
        target_duration = background_audio.duration
        
        if tts_audio.duration > target_duration:
            print(" TTS audio is longer than background, trimming TTS audio...", flush=True)
            tts_audio = tts_audio.subclip(0, target_duration)
        elif tts_audio.duration < target_duration:
            print(" TTS audio is shorter than background, padding TTS with silence...", flush=True)
            silence_duration = target_duration - tts_audio.duration
            silence = create_silence(silence_duration)
            tts_audio = concatenate_audioclips([tts_audio, silence])
            print(f"   Padded TTS audio duration: {tts_audio.duration:.2f} seconds", flush=True)
        
        # Mix the audio - reduce background volume and combine with TTS
        print(" Mixing TTS with background audio...", flush=True)
        background_volume = 0.7  # Lower background volume
        tts_volume = 0.9  # Keep TTS prominent
        
        # Apply volume adjustments and ensure proper fps
        background_adjusted = background_audio.volumex(background_volume)
        tts_adjusted = tts_audio.volumex(tts_volume)
        
        # Get fps from background audio or use standard rate
        audio_fps = getattr(background_audio, 'fps', 44100)
        
        # Ensure both clips have the same fps
        if hasattr(tts_adjusted, 'fps'):
            tts_adjusted.fps = audio_fps
        if hasattr(background_adjusted, 'fps'):
            background_adjusted.fps = audio_fps
        
        # Create composite audio with explicit fps
        try:
            final_audio = CompositeAudioClip([tts_adjusted, background_adjusted])
            final_audio = final_audio.set_duration(target_duration)
            final_audio.fps = audio_fps
            
            print(f"   Final mixed audio duration: {final_audio.duration:.2f} seconds", flush=True)
            
            # Test the composite by trying to get a frame
            try:
                test_frame = final_audio.get_frame(0.1)
                print("   Audio composite created successfully", flush=True)
            except Exception as test_error:
                print(f"   Audio composite test failed: {test_error}", flush=True)
                raise test_error
            
            return final_audio
            
        except Exception as composite_error:
            print(f"   CompositeAudioClip failed: {composite_error}", flush=True)
            print("   Falling back to TTS-only audio...", flush=True)
            # Return just the TTS audio as fallback
            return tts_adjusted
        
    except Exception as e:
        print(f" Error mixing audio: {e}", flush=True)
        print(" Trying fallback: TTS audio only...", flush=True)
        # Fallback to TTS only with proper duration matching
        try:
            if os.path.exists(tts_audio_path):
                fallback_audio = AudioFileClip(tts_audio_path)
                if 'background_audio' in locals() and background_audio:
                    target_duration = background_audio.duration
                    if fallback_audio.duration < target_duration:
                        silence_duration = target_duration - fallback_audio.duration
                        silence = create_silence(silence_duration)
                        fallback_audio = concatenate_audioclips([fallback_audio, silence])
                    elif fallback_audio.duration > target_duration:
                        fallback_audio = fallback_audio.subclip(0, target_duration)
                return fallback_audio
        except Exception as fallback_error:
            print(f" Fallback also failed: {fallback_error}", flush=True)
        return None

def generate_subtitles():
    """Generate SRT subtitles from JSON transcription"""
    try:
        print(" Generating subtitles...", flush=True)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(srt_path)
        if output_dir and not os.path.exists(output_dir):
            print(f" Creating output directory: {output_dir}", flush=True)
            os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(json_path):
            print(f" JSON file not found: {json_path}", flush=True)
            return False
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "speech" not in data:
            print(" No 'speech' key found in JSON data", flush=True)
            return False
            
        subs = pysrt.SubRipFile()
        for i, seg in enumerate(data["speech"]):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]

            sub = pysrt.SubRipItem()
            sub.index = i + 1
            
            # Convert seconds to proper time format
            start_seconds = int(start)
            start_milliseconds = int((start - start_seconds) * 1000)
            end_seconds = int(end)
            end_milliseconds = int((end - end_seconds) * 1000)
            
            # Handle minutes and hours
            sub.start.hours = start_seconds // 3600
            sub.start.minutes = (start_seconds % 3600) // 60
            sub.start.seconds = start_seconds % 60
            sub.start.milliseconds = start_milliseconds
            
            sub.end.hours = end_seconds // 3600
            sub.end.minutes = (end_seconds % 3600) // 60
            sub.end.seconds = end_seconds % 60
            sub.end.milliseconds = end_milliseconds
            
            sub.text = text
            subs.append(sub)

        subs.save(srt_path, encoding='utf-8')
        print(f"Subtitle saved at: {srt_path}", flush=True)
        return True
    except Exception as e:
        print(f"Error generating subtitles: {e}", flush=True)
        return False

def replace_video_audio():
    """Replace video audio with mixed TTS and background audio"""
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}", flush=True)
            return False
        
        print(" Loading video file...", flush=True)
        video = VideoFileClip(video_path)
        print(f"   Video duration: {video.duration:.2f} seconds", flush=True)
        
        # Get mixed audio (TTS + background)
        print(" Creating mixed audio track...", flush=True)
        audio = mix_tts_with_background()
        
        if audio is None:
            print(" Failed to create mixed audio", flush=True)
            return False
        
        print(f"   Mixed audio duration: {audio.duration:.2f} seconds", flush=True)
        
        # Adjust audio duration to match video
        if audio.duration > video.duration:
            print("  Mixed audio is longer than video, trimming audio...", flush=True)
            audio = audio.subclip(0, video.duration)
        elif audio.duration < video.duration:
            print(" Mixed audio is shorter than video, padding with silence...", flush=True)
            # Create silence for the remaining duration
            silence_duration = video.duration - audio.duration
            silence = create_silence(silence_duration)
            
            # Concatenate original audio with silence
            audio = concatenate_audioclips([audio, silence])
            print(f"   Final audio duration: {audio.duration:.2f} seconds", flush=True)
        
        print(" Combining video and audio...", flush=True)
        final_video = video.set_audio(audio)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(" Writing final video...", flush=True)
        final_video.write_videofile(
            output_video_path, 
            codec='libx264', 
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Clean up
        video.close()
        audio.close()
        final_video.close()
        
        print(f" Final video saved at: {output_video_path}", flush=True)
        return True
        
    except Exception as e:
        print(f" Error processing video: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def create_video_with_multiple_audio_tracks():
    """Create a single video with multiple audio tracks using FFmpeg"""
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}", flush=True)
            return False
        
        print(" Loading video file...", flush=True)
        video = VideoFileClip(video_path)
        print(f"   Video duration: {video.duration:.2f} seconds", flush=True)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get mixed audio (TTS + background)
        print(" Creating dubbed audio track...", flush=True)
        dubbed_audio = mix_tts_with_background()
        
        if dubbed_audio is None:
            print(" Failed to create dubbed audio", flush=True)
            video.close()
            return False
        
        # Adjust dubbed audio duration to match video
        if dubbed_audio.duration > video.duration:
            print("  Dubbed audio is longer than video, trimming audio...", flush=True)
            dubbed_audio = dubbed_audio.subclip(0, video.duration)
        elif dubbed_audio.duration < video.duration:
            print(" Dubbed audio is shorter than video, padding with silence...", flush=True)
            silence_duration = video.duration - dubbed_audio.duration
            silence = create_silence(silence_duration)
            dubbed_audio = concatenate_audioclips([dubbed_audio, silence])
            print(f"   Final dubbed audio duration: {dubbed_audio.duration:.2f} seconds", flush=True)
        
        # Extract original audio from video
        print(" Extracting original audio...", flush=True)
        original_audio = video.audio
        
        # Create temporary audio files
        temp_dir = os.path.dirname(output_video_path)
        temp_original_audio = os.path.join(temp_dir, "temp_original_audio.wav")
        temp_dubbed_audio = os.path.join(temp_dir, "temp_dubbed_audio.wav")
        temp_video_no_audio = os.path.join(temp_dir, "temp_video_no_audio.mp4")
        
        print(" Saving temporary audio files...", flush=True)
        original_audio.write_audiofile(temp_original_audio, verbose=False, logger=None)
        dubbed_audio.write_audiofile(temp_dubbed_audio, verbose=False, logger=None)
        
        print(" Saving temporary video without audio...", flush=True)
        video_no_audio = video.without_audio()
        video_no_audio.write_videofile(temp_video_no_audio, verbose=False, logger=None)
        
        # Use FFmpeg to combine video with multiple audio tracks
        print(" Combining video with multiple audio tracks using FFmpeg...", flush=True)
        
        import subprocess
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', temp_video_no_audio,      # Input video (no audio)
            '-i', temp_original_audio,       # Input audio track 1 (original)
            '-i', temp_dubbed_audio,         # Input audio track 2 (dubbed)
            '-c:v', 'copy',                  # Copy video codec (no re-encoding)
            '-c:a', 'aac',                   # Audio codec
            '-map', '0:v',                   # Map video from first input
            '-map', '1:a',                   # Map audio from second input (original)
            '-map', '2:a',                   # Map audio from third input (dubbed)
            '-metadata:s:a:0', 'title=Original Audio',
            '-metadata:s:a:0', 'language=und',
            '-metadata:s:a:1', 'title=Dubbed Audio',
            '-metadata:s:a:1', 'language=und',
            output_video_path
        ]
        
        print(f"   Running FFmpeg command...", flush=True)
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Successfully created video with multiple audio tracks!", flush=True)
            print(f"   Output: {output_video_path}", flush=True)
            print("   Audio Track 1: Original Audio", flush=True)
            print("   Audio Track 2: Dubbed Audio", flush=True)
            print("   Use your media player's audio track selector to switch between them", flush=True)
        else:
            print(f" FFmpeg error: {result.stderr}", flush=True)
            return False
        
        # Clean up temporary files
        print(" Cleaning up temporary files...", flush=True)
        for temp_file in [temp_original_audio, temp_dubbed_audio, temp_video_no_audio]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Clean up MoviePy objects
        video.close()
        video_no_audio.close()
        original_audio.close()
        dubbed_audio.close()
        
        return True
        
    except Exception as e:
        print(f" Error creating video with multiple audio tracks: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the video processing"""
    print(" Starting video processing...", flush=True)
    
    # Generate subtitles
    if generate_subtitles():
        print(" Subtitles generated successfully", flush=True)
    else:
        print("Failed to generate subtitles", flush=True)
    
    # Create video with multiple audio tracks
    print(" Creating video with multiple audio tracks", flush=True)
    if create_video_with_multiple_audio_tracks():
        print(" Video with multiple audio tracks created successfully", flush=True)
    else:
        print(" Failed to create video with multiple audio tracks", flush=True)

if __name__ == "__main__":
    main()