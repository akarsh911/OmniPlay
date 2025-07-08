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
import sys
import shutil
if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding='utf-8')
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
        
        # Intermediate file paths - using global temp folder structure
        self.temp_dir = self.project_root / "temp"
        self.step1_dir = self.temp_dir / "step1"
        self.step2_dir = self.temp_dir / "step2"
        self.step3_dir = self.temp_dir / "step3"
        
        self.extracted_audio_dir = self.step1_dir
        self.vocals_file = None  # Will be set after extraction
        self.transcription_file = self.step2_dir / "output.json"
        self.final_mix_file = self.step3_dir / "final_mix.wav"
        
        # OS detection
        self.is_windows = platform.system() == "Windows"
        
    def print_header(self, step, title):
        """Print a formatted header for each step"""
        print("\n" + "="*80, flush=True)
        print(f"STEP {step}: {title}", flush=True)
        print("="*80, flush=True)
        
    def print_success(self, message):
        """Print success message"""
        print(f"[success] {message}", flush=True)
        
    def print_error(self, message):
        """Print error message"""
        print(f"[error] {message}", flush=True)
        
    def print_info(self, message):
        """Print info message"""
        print(f"ℹ️  {message}", flush=True)
        
    def run_command_in_env(self, env_path, working_dir, command, description):
        """Run a command within a specific virtual environment"""
        self.print_info(f"Running: {description}")
        self.print_info(f"Working directory: {working_dir}")
        self.print_info(f"Environment: {env_path}")
 
        #here we check if the environment path exists
        if self.is_windows:
            activate_script = env_path / "Scripts" / "activate.bat"
            if not activate_script.exists():
                raise FileNotFoundError(f"Environment activation script not found: {activate_script}")

            # Wrap the whole sequence in a single cmd session using `cmd /c`
            full_command = f'cmd /c "call {activate_script} && python -V && where python && {command}"'

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
                print("Environment info:\n", "\n".join(result.stdout.splitlines()[:3]), flush=True)  # Print version + python path
                print("Output:\n", result.stdout, flush=True)
                return result

            except subprocess.CalledProcessError as e:
                self.print_error(f"{description} failed")
                print("Environment output:\n", e.stdout, flush=True)
                print("Error output:\n", e.stderr, flush=True)
                print("Command output:\n", e.stdout, flush=True)
                raise

        else:
            activate_script = env_path / "bin" / "activate"
            if not activate_script.exists():
                raise FileNotFoundError(f"Environment activation script not found: {activate_script}")

            full_command = f'source "{activate_script}" && python3 -V && which python3 && {command}'

            try:
                result = subprocess.run(
                    full_command,
                    shell=True,
                    cwd=working_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    executable='/bin/bash'
                )
                self.print_success(f"{description} completed successfully")
                print("Environment info:\n", result.stdout.splitlines()[1], flush=True)  # Show python path
                print("Output:\n", result.stdout, flush=True)
                return result

            except subprocess.CalledProcessError as e:
                self.print_error(f"{description} failed")
                print("Error output:\n", e.stderr, flush=True)
                print("Command output:\n", e.stdout, flush=True)
                raise

    def run_command_native(self, working_dir, command, description):
        """Run a command in the native environment (no virtual environment)"""
        self.print_info(f"Running: {description}")
        self.print_info(f"Working directory: {working_dir}")
        self.print_info("Environment: Native (no virtual environment)")
        
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
            print("Command output:", e.stdout, flush=True)
            raise

    def validate_environment(self, env_path, required_packages):
        """Validate that required packages are installed in the environment"""
        self.print_info(f"Validating environment: {env_path}")
        
        if self.is_windows:
            activate_script = env_path / "Scripts" / "activate.bat"
            python_exe = env_path / "Scripts" / "python.exe"
        else:
            activate_script = env_path / "bin" / "activate"
            python_exe = env_path / "bin" / "python"
        
        if not activate_script.exists():
            self.print_error(f"Environment activation script not found: {activate_script}")
            return False
            
        if not python_exe.exists():
            self.print_error(f"Python executable not found: {python_exe}")
            return False
        
        # Test package imports
        for package in required_packages:
            test_command = f'"{python_exe}" -c "import {package}; print(f\\"✅ {package} available\\")"'
            try:
                result = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.print_success(f"Package {package} is available")
            except subprocess.CalledProcessError:
                self.print_error(f"Package {package} is NOT available in {env_path}")
                return False
        
        return True

    def run_command_in_env_with_fallback(self, env_path, working_dir, command, description, required_packages=None, allow_native_fallback=True):
        """Run command in environment with validation and fallback to native"""
        
        # First validate the environment if required packages are specified
        if required_packages:
            if not self.validate_environment(env_path, required_packages):
                if allow_native_fallback:
                    self.print_info("Environment validation failed, trying native environment...")
                    return self.run_command_native(working_dir, command, f"{description} (fallback to native)")
                else:
                    raise EnvironmentError(f"Required packages not available in {env_path}")
        
        # Try to run in the virtual environment first
        try:
            return self.run_command_in_env(env_path, working_dir, command, description)
        except subprocess.CalledProcessError as e:
            if allow_native_fallback:
                self.print_error(f"Virtual environment execution failed: {e}")
                self.print_info("Falling back to native environment...")
                return self.run_command_native(working_dir, command, f"{description} (fallback to native)")
            else:
                raise

    def step1_extract_audio(self):
        """Step 1: Extract audio from video using VideoSplitter"""
        self.print_header(1, "AUDIO EXTRACTION")
        
        # Check if input video exists
        if not self.input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        
        # Check if extract script exists
        if not self.extract_script.exists():
            raise FileNotFoundError(f"Extract script not found: {self.extract_script}")
        
        # Run extraction
        working_dir = self.project_root
        command = f'python "{self.extract_script}" "{self.input_video_path}"'
        
        self.run_command_in_env(
            self.video_splitter_env,
            working_dir,
            command,
            "Audio extraction from video"
        )
        
        # Find the extracted vocals file
        if self.extracted_audio_dir.exists():
            vocals_files = list(self.extracted_audio_dir.glob("*_vocals.wav"))
            if vocals_files:
                self.vocals_file = vocals_files[0]
                self.print_success(f"Found extracted vocals: {self.vocals_file}")
            else:
                raise FileNotFoundError("No vocals file found after extraction")
        else:
            raise FileNotFoundError("Extracted audio directory not found")

    def step2_transcribe_audio(self):
        """Step 2: Transcribe audio using VoiceTranscriber"""
        self.print_header(2, "AUDIO TRANSCRIPTION")
        
        if not self.vocals_file or not self.vocals_file.exists():
            raise FileNotFoundError("Vocals file not found from previous step")
        
        # Check if transcriber script exists
        if not self.transcriber_script.exists():
            raise FileNotFoundError(f"Transcriber script not found: {self.transcriber_script}")
        
        # Run transcription
        working_dir = self.project_root
        command = f'python "{self.transcriber_script}" "{self.vocals_file}" "{self.transcription_file}"'
        
        self.run_command_in_env(
            self.voice_transcriber_env,
            working_dir,
            command,
            "Audio transcription and speaker diarization"
        )
        
        # Check if transcription file was created
        if not self.transcription_file.exists():
            raise FileNotFoundError(f"Transcription file not created: {self.transcription_file}")
        
        self.print_success(f"Transcription completed: {self.transcription_file}")

    # def step3_clone_voices(self):
    #     """Step 3: Clone voices using VoiceCloner"""
    #     self.print_header(3, "VOICE CLONING")
        
    #     if not self.transcription_file.exists():
    #         raise FileNotFoundError("Transcription file not found from previous step")
        
    #     # Check if cloner script exists
    #     if not self.cloner_script.exists():
    #         raise FileNotFoundError(f"Voice cloner script not found: {self.cloner_script}")
        
    #     # Run voice cloning
    #     working_dir = self.project_root
    #     command = f'python {self.cloner_script} '
        
    #     self.run_command_in_env(
    #         self.voice_cloner_env,
    #         working_dir,
    #         command,
    #         "Voice cloning and synthesis"
    #     )
        
    #     # Check for output files
    #     output_dir = self.project_root / "VoiceCloner" / "output"
    #     if output_dir.exists():
    #         output_files = list(output_dir.glob("*.wav"))
    #         if output_files:
    #             self.print_success(f"Voice cloning completed. Output files in: {output_dir}")
    #             for file in output_files:
    #                 self.print_info(f"Generated: {file.name}")
    #         else:
    #             self.print_error("No output files found after voice cloning")
    #     else:
    #         self.print_error("Output directory not found after voice cloning")
    def step3_clone_voices(self):
        """Step 3: Clone voices using VoiceCloner"""
        self.print_header(3, "VOICE CLONING")

        if not self.transcription_file.exists():
            raise FileNotFoundError("Transcription file not found from previous step")

        if not self.cloner_script.exists():
            raise FileNotFoundError(f"Voice cloner script not found: {self.cloner_script}")

        # Construct correct Python path inside virtualenv
        if self.is_windows:
            python_exec = self.voice_cloner_env / "Scripts" / "python.exe"
        else:
            python_exec = self.voice_cloner_env / "bin" / "python"

        if not python_exec.exists():
            raise FileNotFoundError(f"Python executable not found in env: {python_exec}")

        # Build command using correct interpreter
        working_dir = self.project_root
        command = f'"{python_exec}" "{self.cloner_script}"'

        self.print_info(f"Running voice cloning with: {python_exec}")
        self.print_info(f"Command: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True
            )
            self.print_success("Voice cloning and synthesis completed successfully")
            print("Output:\n", result.stdout)

            # Check for output files
            output_dir = self.project_root / "temp" / "step3"
            if output_dir.exists():
                output_files = list(output_dir.glob("*.wav"))
                if output_files:
                    self.print_success(f"Voice cloning completed. Output files in: {output_dir}")
                    for file in output_files:
                        self.print_info(f"Generated: {file.name}")
                else:
                    self.print_error("No output files found after voice cloning")
            else:
                self.print_error("Output directory not found after voice cloning")

        except subprocess.CalledProcessError as e:
            self.print_error("Voice cloning and synthesis failed")
            print("Command output:\n", e.stdout)
            print("Error output:\n", e.stderr)
            raise

    def step4_mix_final_video(self, output_video_path=None, srt_path=None):
        """Step 4: Mix final video using mixer.py"""
        self.print_header(4, "FINAL VIDEO MIXING")
        
        
        # Check if mixer script exists
        if not self.mixer_script.exists():
            raise FileNotFoundError(f"Mixer script not found: {self.mixer_script}")
        
        # Check if required input files exist
        required_files = [
            self.transcription_file,  # temp/step2/output.json
            self.final_mix_file       # temp/step3/final_mix.wav
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                self.print_error(f"Required file not found: {file_path}")
                self.print_info("This file should have been generated in previous steps")
        
        # Run mixer with root environment and fallback to native
        working_dir = self.project_root
        command = f'python "{self.mixer_script}" "{self.input_video_path}"'
        
        self.run_command_in_env_with_fallback(
            self.root_env,
            working_dir,
            command,
            "Final video mixing with synthesized audio",
            required_packages=[],  # Validate moviepy is available
            allow_native_fallback=True
        )
        
        # Check for output files
        final_video = Path(output_video_path)
        subtitles_file = Path(srt_path)
        
        if final_video.exists():
            self.print_success(f"Final video created: {final_video}")
        else:
            self.print_error("Final video file not found after mixing")
            
        if subtitles_file.exists():
            self.print_success(f"Subtitles generated: {subtitles_file}")
        else:
            self.print_error("Subtitle file not found after mixing")

    def run_pipeline(self):
        """Run the complete video translation pipeline"""
        start_time = time.time()
        
        print("🎬 VIDEO TRANSLATION PIPELINE STARTING", flush=True)
        print(f"📁 Input video: {self.input_video_path}", flush=True)
        print(f"📁 Project root: {self.project_root}", flush=True)
        
        try:
            # Step 1: Extract audio
            self.step1_extract_audio()
            
            # Step 2: Transcribe audio
            self.step2_transcribe_audio()
            
            # Step 3: Clone voices
            self.step3_clone_voices()
            
            # Step 4: Mix final video
            self.step4_mix_final_video()
            
            # Final success message
            end_time = time.time()
            duration = end_time - start_time
            # Delete temp folder on success
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.print_success(f"Cleaned up temporary files: {self.temp_dir}")
            print("\n" + "="*80, flush=True)
            print("[comlete] VIDEO TRANSLATION PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
            print("="*80, flush=True)
            print(f"⏱️  Total time: {duration:.2f} seconds", flush=True)
            print(f"📁 Check the following directories for outputs:", flush=True)
            print(f"   • VideoSplitter/extracted_audio/ - Extracted audio files", flush=True)
            print(f"   • VoiceTranscriber/ - Transcription and diarization", flush=True)
            print(f"   • VoiceCloner/output/ - Synthesized audio", flush=True)
            print(f"   • final_video.mp4 - Complete translated video", flush=True)
            print(f"   • final_subtitles.srt - Generated subtitles", flush=True)
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n" + "="*80, flush=True)
            print("💥 PIPELINE FAILED", flush=True)
            print("="*80, flush=True)
            print(f"[error] Error: {str(e)}", flush=True)
            print(f"⏱️  Time before failure: {duration:.2f} seconds", flush=True)
            
                  
            raise


def main():
    """Main function to handle command line arguments and run the pipeline"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("[error] Usage: python run.py <input_video_path>", flush=True)
        print("Example: python run.py input/input.mp4", flush=True)
        print("Example: python run.py \"C:/path/to/your/video.mp4\"", flush=True)
        sys.exit(1)
    
    # Get input video path
    input_video_path = sys.argv[1]
    
    # Validate input
    if not os.path.exists(input_video_path):
        print(f"[error] Input video file not found: {input_video_path}", flush=True)
        print("Please ensure your video file exists at the specified path.", flush=True)
        sys.exit(1)
    
    # Initialize and run pipeline
    try:
        pipeline = VideoTranslationPipeline(input_video_path)
        pipeline.run_pipeline()
        
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {str(e)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
