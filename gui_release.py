import os
import sys
import time
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # PyInstaller extracted path
    return os.path.dirname(os.path.abspath(__file__))


class PipelineRunner(QThread):
    log_output = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, float)  # success, elapsed_time

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.process = None

    def run(self):
        start_time = time.time()
        success = False
        try:
            run_py = os.path.join(get_base_path(), "runner.py")
            self.process = subprocess.Popen(
                [sys.executable, run_py, self.video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )

            step_progress = {
                "STEP 1": 25,
                "STEP 2": 50,
                "STEP 3": 75,
                "STEP 4": 90,
                "VIDEO TRANSLATION PIPELINE COMPLETED": 100
            }

            for line in self.process.stdout:
                self.log_output.emit(line)
                for step, progress in step_progress.items():
                    if step in line:
                        self.progress_update.emit(progress)

            self.process.wait()
            if self.process.returncode == 0:
                success = True

        except Exception as e:
            self.log_output.emit(f"[ERROR] {str(e)}")

        elapsed = time.time() - start_time
        self.finished.emit(success, elapsed)

    def terminate(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass
        super().terminate()


class VideoTranslatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Translator")
        self.resize(600, 600)
        self.layout = QVBoxLayout()

        self.label = QLabel("Browse")
        self.input_path = QLineEdit()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.browse_file)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input_path)
        self.layout.addWidget(self.open_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet("font-size: 18px; padding: 10px;")
        self.start_btn.clicked.connect(self.start_pipeline)
        self.layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_pipeline)
        self.layout.addWidget(self.cancel_btn)

        self.progress_label = QLabel("Progress")
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)

        self.time_label = QLabel("⏱ Elapsed: 0.00 sec")
        self.layout.addWidget(self.time_label)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Terminal output")
        self.layout.addWidget(self.output_box)

        self.setLayout(self.layout)

        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.start_time = None
        self.runner = None

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi)")
        if file_path:
            self.input_path.setText(file_path)
            self.setWindowTitle(f"Video Translator - {os.path.basename(file_path)}")

    def reset_ui(self):
        self.progress_bar.setValue(0)
        self.time_label.setText("⏱ Elapsed: 0.00 sec")
        self.output_box.clear()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.setWindowTitle("Video Translator")

    def start_pipeline(self):
        video_file = self.input_path.text().strip()
        if not video_file or not os.path.exists(video_file):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid video file.")
            return

        self.reset_ui()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.start_time = time.time()
        self.elapsed_timer.start(1000)

        self.runner = PipelineRunner(video_file)
        self.runner.log_output.connect(self.append_output)
        self.runner.progress_update.connect(self.progress_bar.setValue)
        self.runner.finished.connect(self.pipeline_finished)
        self.runner.start()

    def cancel_pipeline(self):
        if self.runner and self.runner.isRunning():
            self.runner.terminate()
            self.output_box.append("[CANCELLED] Pipeline was cancelled by user.")
            self.pipeline_finished(False, time.time() - self.start_time)

    def update_elapsed_time(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.time_label.setText(f"⏱ Elapsed: {elapsed:.2f} sec")

    def append_output(self, text):
        self.output_box.append(text)
        self.output_box.ensureCursorVisible()

    def pipeline_finished(self, success, elapsed_time):
        self.elapsed_timer.stop()
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            QMessageBox.information(self, "Success", f"Pipeline completed in {elapsed_time:.2f} seconds.")
        else:
            QMessageBox.critical(self, "Error", f"Pipeline failed or cancelled after {elapsed_time:.2f} seconds.")

        self.setWindowTitle("Video Translator")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTranslatorUI()
    window.show()
    sys.exit(app.exec_())
