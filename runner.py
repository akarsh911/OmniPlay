#!/usr/bin/env python3
"""
Video Translation Pipeline Runner
Orchestrates the complete video translation process across multiple environments.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import time
import shutil

if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')
os.environ["PATH"] += os.pathsep + r"C:\Users\Akars\ffmpeg\ffmpeg-2025-06-02-git-688f3944ce-full_build\bin"

class VideoTranslationPipeline:
    def __init__(self, input_video_path):
        self.input_video_path = Path(input_video_path).resolve()
        self.project_root = Path(__file__).parent.resolve()
        
        # Environment paths
        self.video_splitter_env = self.project_root / "VideoSplitter" / "env388"
        self.voice_transcriber_env = self.project_root / "VoiceTranscriber" / "env309"
        self.voice_cloner_env = self.project_root / "VoiceCloner" / "env310"
        self.root_env = self.project_root / "envf"  # Root environment for mixer
        
        # Script paths
        self.extract_script = self.project_root / "VideoSplitter" / "extract_v0.py"
        self.transcriber_script = self.project_root / "VoiceTranscriber" / "v4.py"
        self.cloner_script = self.project_root / "VoiceCloner" / "v7.py"
        self.mixer_script = self.project_root / "mixer.py"
        
        # Intermediate file paths
        self.temp_dir = self.project_root / "temp"
        self.step1_dir = self.temp_dir / "step1"
        self.step2_dir = self.temp_dir / "step2"
        self.step3_dir = self.temp_dir / "step3"
        
        self.extracted_audio_dir = self.step1_dir
        self.vocals_file = None
        self.transcription_file = self.step2_dir / "output.json"
        self.final_mix_file = self.step3_dir / "final_mix.wav"
        
        self.is_windows = platform.system() == "Windows"

    def print_header(self, step, title):
        print("\n" + "="*80, flush=True)
        print(f"STEP {step}: {title}", flush=True)
        print("="*80, flush=True)

    def print_success(self, message):
        print(f"[success] {message}", flush=True)

    def print_error(self, message):
        print(f"[error] {message}", flush=True)

    def print_info(self, message):
        print(f"ℹ️  {message}", flush=True)

    def run_command_in_env(self, env_path, working_dir, command, description):
        """Run a command within a specific virtual environment using direct python.exe path"""
        self.print_info(f"Running: {description}")
        self.print_info(f"Working directory: {working_dir}")
        self.print_info(f"Environment: {env_path}")

        python_exe = env_path / ("Scripts" if self.is_windows else "bin") / "python.exe"
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found in: {python_exe}")

        split_command = command.split()
        split_command[0] = f'"{str(python_exe)}"'
        full_command = " ".join(split_command)

        try:
            result = subprocess.run(
                full_command,
                shell=True,
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True
            )
            self.print_success(f"{description} completed successfully")
            print("Output:\n", result.stdout, flush=True)
            return result

        except subprocess.CalledProcessError as e:
            self.print_error(f"{description} failed")
            print("Error output:\n", e.stderr, flush=True)
            raise

    def run_command_native(self, working_dir, command, description):
        self.print_info(f"Running: {description}")
        self.print_info(f"Working directory: {working_dir}")
        self.print_info("Environment: Native")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True
            )
            self.print_success(f"{description} completed successfully")
            print("Output:", result.stdout, flush=True)
            return result

        except subprocess.CalledProcessError as e:
            self.print_error(f"{description} failed")
            print("Error output:", e.stderr, flush=True)
            raise

    def run_command_in_env_with_fallback(self, env_path, working_dir, command, description):
        try:
            return self.run_command_in_env(env_path, working_dir, command, description)
        except Exception as e:
            self.print_error(f"Virtual environment failed: {e}")
            self.print_info("Falling back to native environment")
            return self.run_command_native(working_dir, command, f"{description} (native fallback)")

    def step1_extract_audio(self):
        self.print_header(1, "AUDIO EXTRACTION")

        if not self.input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        if not self.extract_script.exists():
            raise FileNotFoundError(f"Extract script not found: {self.extract_script}")

        command = f'python "{self.extract_script}" "{self.input_video_path}"'
        self.run_command_in_env(self.video_splitter_env, self.project_root, command, "Audio extraction")

        if self.extracted_audio_dir.exists():
            vocals = list(self.extracted_audio_dir.glob("*_vocals.wav"))
            if vocals:
                self.vocals_file = vocals[0]
                self.print_success(f"Extracted vocals found: {self.vocals_file}")
            else:
                raise FileNotFoundError("No vocals file found")
        else:
            raise FileNotFoundError("Extracted audio folder missing")

    def step2_transcribe_audio(self):
        self.print_header(2, "AUDIO TRANSCRIPTION")

        if not self.vocals_file or not self.vocals_file.exists():
            raise FileNotFoundError("Vocals file not found")

        if not self.transcriber_script.exists():
            raise FileNotFoundError(f"Transcriber script missing: {self.transcriber_script}")

        command = f'python "{self.transcriber_script}" "{self.vocals_file}" "{self.transcription_file}"'
        self.run_command_in_env(self.voice_transcriber_env, self.project_root, command, "Transcription")

        if not self.transcription_file.exists():
            raise FileNotFoundError("Transcription output not found")

    def step3_clone_voices(self):
        self.print_header(3, "VOICE CLONING")

        if not self.transcription_file.exists():
            raise FileNotFoundError("Missing transcription output")

        if not self.cloner_script.exists():
            raise FileNotFoundError("Cloner script missing")

        command = f'python "{self.cloner_script}"'
        self.run_command_in_env(self.voice_cloner_env, self.project_root, command, "Voice Cloning")

        output_dir = self.step3_dir
        if output_dir.exists():
            files = list(output_dir.glob("*.wav"))
            if files:
                self.print_success("Voice cloning output:")
                for f in files:
                    print(f"🗣️  {f.name}")
            else:
                self.print_error("No voice cloning output found")
        else:
            self.print_error("Voice cloning directory not created")

    def step4_mix_final_video(self):
        self.print_header(4, "FINAL VIDEO MIXING")

        if not self.mixer_script.exists():
            raise FileNotFoundError("Mixer script missing")
        if not self.transcription_file.exists():
            raise FileNotFoundError("Missing transcription file")
        if not self.final_mix_file.exists():
            raise FileNotFoundError("Missing final mix audio")

        command = f'python "{self.mixer_script}" "{self.input_video_path}"'
        self.run_command_in_env_with_fallback(self.root_env, self.project_root, command, "Mixing final video")

    def run_pipeline(self):
        start_time = time.time()

        print("🎬 VIDEO TRANSLATION PIPELINE STARTING")
        print(f"📁 Input video: {self.input_video_path}")
        print(f"📁 Project root: {self.project_root}")

        try:
            self.step1_extract_audio()
            self.step2_transcribe_audio()
            self.step3_clone_voices()
            self.step4_mix_final_video()

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.print_success("Temporary files cleaned")

            print("\n✅ VIDEO TRANSLATION PIPELINE COMPLETED")
            print(f"📁 Output video: final_video.mp4")
            print(f"📄 Subtitles: final_subtitles.srt")
            print(f"⏱️ Duration: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            print("\n❌ PIPELINE FAILED")
            print(f"[error] {str(e)}")
            print(f"⏱️ Time before failure: {time.time() - start_time:.2f} seconds")
            raise


def main():
    if len(sys.argv) < 2:
        print("[error] Usage: python run.py <input_video_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"[error] File not found: {input_path}")
        sys.exit(1)

    try:
        pipeline = VideoTranslationPipeline(input_path)
        pipeline.run_pipeline()
    except Exception as e:
        print(f"[fatal] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
