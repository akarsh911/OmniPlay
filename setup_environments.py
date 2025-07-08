#!/usr/bin/env python3
"""
VideoTranslator Environment Setup Script
Automatically sets up all required Python environments and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description="", check=True, cwd=None):
    """Run a command with error handling and logging."""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, cwd=cwd, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, cwd=cwd,
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   stderr: {e.stderr.strip()}")
        return False

def check_python_version(python_cmd, required_version):
    """Check if Python version meets requirements."""
    try:
        result = subprocess.run([python_cmd, "--version"], 
                              capture_output=True, text=True, check=True)
        version = result.stdout.strip().split()[1]
        major, minor = map(int, version.split('.')[:2])
        req_major, req_minor = map(int, required_version.split('.'))
        
        if major == req_major and minor == req_minor:
            print(f"   ✅ Found {python_cmd}: {version}")
            return True
        else:
            print(f"   ❌ {python_cmd} version {version} doesn't match required {required_version}")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"   ❌ {python_cmd} not found")
        return False

def setup_environment(env_path, python_cmd, requirements_file, description):
    """Set up a Python virtual environment."""
    print(f"\n🔨 Setting up {description}")
    
    # Create virtual environment
    if not env_path.exists():
        if not run_command([python_cmd, "-m", "venv", str(env_path)], 
                          f"Creating virtual environment at {env_path}"):
            return False
    else:
        print(f"   ✅ Environment already exists at {env_path}")
    
    # Get activation script path
    if platform.system() == "Windows":
        activate_script = env_path / "Scripts" / "activate.bat"
        pip_cmd = str(env_path / "Scripts" / "pip.exe")
    else:
        activate_script = env_path / "bin" / "activate"
        pip_cmd = str(env_path / "bin" / "pip")
    
    # Install requirements
    if requirements_file.exists():
        if not run_command([pip_cmd, "install", "-r", str(requirements_file)],
                          f"Installing requirements from {requirements_file.name}"):
            return False
    else:
        print(f"   ⚠️ Requirements file not found: {requirements_file}")
    
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print("\n🎬 Checking FFmpeg installation...")
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, check=True)
        version_line = result.stdout.split('\n')[0]
        print(f"   ✅ {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ❌ FFmpeg not found")
        return False

def install_ffmpeg_instructions():
    """Provide FFmpeg installation instructions."""
    system = platform.system()
    print("\n📋 FFmpeg Installation Instructions:")
    
    if system == "Windows":
        print("   1. Download FFmpeg from: https://ffmpeg.org/download.html")
        print("   2. Extract to C:\\ffmpeg")
        print("   3. Add C:\\ffmpeg\\bin to your system PATH")
        print("   4. Restart command prompt/PowerShell")
        print("   Alternative: choco install ffmpeg")
    elif system == "Darwin":  # macOS
        print("   Install using Homebrew:")
        print("   brew install ffmpeg")
    else:  # Linux
        print("   Ubuntu/Debian:")
        print("   sudo apt update && sudo apt install ffmpeg")
        print("   CentOS/RHEL:")
        print("   sudo yum install ffmpeg")

def check_pyqt5():
    """Check if PyQt5 is installed for GUI."""
    print("\n🖥️ Checking PyQt5 for GUI support...")
    try:
        import PyQt5
        print("   ✅ PyQt5 is available")
        return True
    except ImportError:
        print("   ❌ PyQt5 not found")
        return False

def install_pyqt5_instructions():
    """Provide PyQt5 installation instructions."""
    system = platform.system()
    print("\n📋 PyQt5 Installation Instructions:")
    print("   pip install PyQt5")
    
    if system == "Linux":
        print("   If pip install fails, try:")
        print("   sudo apt install python3-pyqt5 python3-pyqt5-dev")

def main():
    """Main setup function."""
    print("🚀 VideoTranslator Environment Setup")
    print("=" * 50)
    
    project_root = Path(__file__).parent.resolve()
    print(f"📁 Project root: {project_root}")
    
    # Check Python versions
    print("\n🐍 Checking Python installations...")
    python_versions = {
        "3.8": "python3.8" if platform.system() != "Windows" else "python",
        "3.9": "python3.9" if platform.system() != "Windows" else "python", 
        "3.10": "python3.10" if platform.system() != "Windows" else "python"
    }
    
    python_found = {}
    for version, cmd in python_versions.items():
        if platform.system() == "Windows":
            # On Windows, try py launcher first
            py_cmd = f"py -{version}"
            if check_python_version(py_cmd.split(), version):
                python_found[version] = py_cmd.split()
            elif check_python_version([cmd], version):
                python_found[version] = [cmd]
        else:
            if check_python_version([cmd], version):
                python_found[version] = [cmd]
    
    if len(python_found) < 3:
        missing = set(python_versions.keys()) - set(python_found.keys())
        print(f"\n❌ Missing Python versions: {', '.join(missing)}")
        print("   Please install the required Python versions:")
        print("   - Python 3.8 for VideoSplitter")
        print("   - Python 3.9 for VoiceTranscriber") 
        print("   - Python 3.10 for VoiceCloner and root environment")
        return False
    
    # Set up environments
    environments = [
        {
            "path": project_root / "envf",
            "python": python_found["3.10"],
            "requirements": project_root / "requirements_root.txt",
            "description": "Root Environment (Python 3.10)"
        },
        {
            "path": project_root / "VideoSplitter" / "env388",
            "python": python_found["3.8"],
            "requirements": project_root / "VideoSplitter" / "requirements_splitter.txt",
            "description": "VideoSplitter Environment (Python 3.8)"
        },
        {
            "path": project_root / "VoiceTranscriber" / "env309",
            "python": python_found["3.9"],
            "requirements": project_root / "VoiceTranscriber" / "requirements_transcriber.txt",
            "description": "VoiceTranscriber Environment (Python 3.9)"
        },
        {
            "path": project_root / "VoiceCloner" / "env310",
            "python": python_found["3.10"],
            "requirements": project_root / "VoiceCloner" / "requirements_cloner.txt",
            "description": "VoiceCloner Environment (Python 3.10)"
        }
    ]
    
    success_count = 0
    for env_config in environments:
        if setup_environment(
            env_config["path"],
            env_config["python"][0] if len(env_config["python"]) == 1 else " ".join(env_config["python"]),
            env_config["requirements"],
            env_config["description"]
        ):
            success_count += 1
    
    # Create root requirements file if it doesn't exist
    root_req_file = project_root / "requirements_root.txt"
    if not root_req_file.exists():
        print(f"\n📝 Creating {root_req_file.name}")
        with open(root_req_file, 'w') as f:
            f.write("moviepy==1.0.3\n")
            f.write("pysrt\n")
            f.write("numpy\n")
            f.write("PyQt5\n")
    
    # Check additional dependencies
    ffmpeg_ok = check_ffmpeg()
    pyqt5_ok = check_pyqt5()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)
    print(f"✅ Python environments: {success_count}/4")
    print(f"{'✅' if ffmpeg_ok else '❌'} FFmpeg: {'Available' if ffmpeg_ok else 'Missing'}")
    print(f"{'✅' if pyqt5_ok else '❌'} PyQt5: {'Available' if pyqt5_ok else 'Missing'}")
    
    if success_count == 4 and ffmpeg_ok and pyqt5_ok:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 You can now run:")
        print("   python gui_launcher.py  # GUI mode")
        print("   python run.py video.mp4  # CLI mode")
    else:
        print("\n⚠️ Setup incomplete. Please address the following:")
        
        if success_count < 4:
            print("   - Fix Python environment setup errors above")
        
        if not ffmpeg_ok:
            install_ffmpeg_instructions()
        
        if not pyqt5_ok:
            install_pyqt5_instructions()
    
    return success_count == 4 and ffmpeg_ok and pyqt5_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
