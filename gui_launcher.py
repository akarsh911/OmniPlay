import os
import sys
import time
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QMessageBox
)
from PyQt5.QtCore import QTimer

import sys
import os
import subprocess
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class PipelineRunner(QThread):
    log_output = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, float)  # success, elapsed_time

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        start_time = time.time()
        success = False

        try:
            process = subprocess.Popen(
                [sys.executable, "run.py", self.video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                encoding="utf-8",  # 👈 Force UTF-8 decoding
                errors="replace"   # 👈 Replace problematic characters to avoid crashing
            )


            step_progress = {
                "STEP 1": 25,
                "STEP 2": 50,
                "STEP 3": 75,
                "STEP 4": 90,
                "VIDEO TRANSLATION PIPELINE COMPLETED": 100
            }

            for line in process.stdout:
                self.log_output.emit(line)
                for step, progress in step_progress.items():
                    if step in line:
                        self.progress_update.emit(progress)

            process.wait()
            if process.returncode == 0:
                success = True

        except Exception as e:
            self.log_output.emit(f"[ERROR] {str(e)}")

        elapsed = time.time() - start_time
        self.finished.emit(success, elapsed)

class VideoTranslatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Translator")
        self.resize(600, 600)
        # self.setFixedSize(600, 600)

        self.layout = QVBoxLayout()

        # Browse
        self.label = QLabel("Browse")
        self.input_path = QLineEdit()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.browse_file)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input_path)
        self.layout.addWidget(self.open_btn)

        # Start
        self.start_btn = QPushButton("start")
        self.start_btn.setStyleSheet("font-size: 24px; padding: 12px;")
        self.start_btn.clicked.connect(self.start_pipeline)
        self.layout.addWidget(self.start_btn)

        # Progress bar
        self.progress_label = QLabel("Progress")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)

        # Elapsed time
        self.time_label = QLabel("⏱ Elapsed: 0.00 sec")
        self.layout.addWidget(self.time_label)

        # Terminal output
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("terminal output")
        self.layout.addWidget(self.output_box)

        self.setLayout(self.layout)

        # Timer for elapsed time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.start_time = None

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi)")
        if file_path:
            self.input_path.setText(file_path)

    def start_pipeline(self):
        video_file = self.input_path.text().strip()

        if not video_file or not os.path.exists(video_file):
            QMessageBox.warning(self, "Invalid Input", "Please select a valid video file.")
            return

        self.output_box.clear()
        self.progress_bar.setValue(0)
        self.time_label.setText("⏱ Elapsed: 0.00 sec")
        self.start_btn.setEnabled(False)

        self.runner = PipelineRunner(video_file)
        self.runner.log_output.connect(self.append_output)
        self.runner.progress_update.connect(self.progress_bar.setValue)
        self.runner.finished.connect(self.pipeline_finished)
        self.runner.start()

        self.start_time = time.time()
        self.elapsed_timer.start(1000)  # update every 1 sec

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
        if success:
            QMessageBox.information(self, "Success", f"Pipeline completed in {elapsed_time:.2f} seconds.")
        else:
            QMessageBox.critical(self, "Error", f"Pipeline failed after {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTranslatorUI()
    window.show()
    sys.exit(app.exec_())