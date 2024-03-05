import sys
import json
import pandas as pd
import os
import serial
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
    QSpinBox,  # Import QSpinBox for the time duration setting
    QLabel,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from joblib import load

script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = f"{script_dir}/trained_model.joblib"

default_directory = "/home/parker/Dev/GitHub/PETEr/subsystems/spectrophotometer/spectralDataClassifier/saved_testing_data"
default_filename = "output.txt"
default_filepath = os.path.join(default_directory, default_filename)

if not os.path.exists(default_directory):
    os.makedirs(default_directory)


class DataThread(QThread):
    data_signal = pyqtSignal(str)
    stop_signal = pyqtSignal()

    def __init__(self, arduino, filename, model, duration):
        super(DataThread, self).__init__()
        self.arduino = arduino
        self.filename = filename
        self.model = model
        self.duration = duration
        self.is_running = True
        self.file = None
        self.open_file()
        self.elapsed_time = 0

    def open_file(self):
        if self.file is not None:
            self.file.close()
        self.file = open(self.filename, "a")

    def run(self):
        timer = QTimer()
        timer.timeout.connect(self.stop)
        timer.timeout.connect(self.update_time_remaining)
        timer.start(1000)  # Update every second

        while self.is_running and self.elapsed_time < self.duration:
            data = self.arduino.readline()
            if data:
                decoded_data = data.decode("utf-8", errors="ignore").strip()
                self.process_data(decoded_data)
                self.elapsed_time += 1

        timer.stop()

    def update_time_remaining(self):
        time_remaining = max(0, self.duration - self.elapsed_time)
        self.emit_signal(time_remaining)

    def emit_signal(self, time_remaining):
        self.stop_signal.emit() if time_remaining == 0 else None

    def stop(self):
        self.is_running = False
        if self.file:
            self.file.close()
        self.wait()

    def process_data(self, decoded_data):
        try:
            if "{" in decoded_data and "}" in decoded_data:
                data_dict = json.loads(decoded_data)
                if len(data_dict) == 18:
                    data_df = pd.DataFrame([data_dict])
                    prediction = self.model.predict_proba(data_df)[0][1]
                    result = f"Prediction: {prediction*100:.3f}%"
                    self.data_signal.emit(result)
                    self.file.write(f"Sample: {json.dumps(data_dict)} - {result}\n")
        except Exception as e:
            print(f"Error processing data: {e}")


class MainWindow(QMainWindow):
    def __init__(self, model):
        super(MainWindow, self).__init__()
        self.model = model
        self.current_file_path = default_filepath
        self.duration = 2  # Default duration is 2 minutes
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Live Predictions")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_recording)
        layout.addWidget(self.stop_button)

        self.change_save_location_button = QPushButton("Change Save Location")
        self.change_save_location_button.clicked.connect(self.change_save_location)
        layout.addWidget(self.change_save_location_button)

        # Spin box for adjusting the recording duration
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setMinimum(1)  # Minimum duration is 1 minute
        self.duration_spinbox.setMaximum(10)  # Maximum duration is 10 minutes
        self.duration_spinbox.setValue(self.duration)  # Default duration is 2 minutes
        self.duration_spinbox.valueChanged.connect(self.duration_changed)
        layout.addWidget(QLabel("Recording Duration (Minutes):"))
        layout.addWidget(self.duration_spinbox)

        self.time_label = QLabel("Time Remaining: --")  # Set initial text
        layout.addWidget(self.time_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.arduino = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
        self.data_thread = None  # Initialized on start

    def start_recording(self):
        if self.data_thread is None or not self.data_thread.isRunning():
            self.duration = self.duration_spinbox.value() * 60
            self.data_thread = DataThread(
                self.arduino, self.current_file_path, self.model, self.duration
            )
            self.data_thread.data_signal.connect(self.update_text)
            self.data_thread.stop_signal.connect(self.recording_stopped)
            self.data_thread.start()
            self.update_time_remaining()  # Initialize the time label

    def stop_recording(self):
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()

    def update_text(self, text):
        self.text_edit.append(text)

    def change_save_location(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select New File Location",
            self.current_file_path,
            "All Files (*)",
            options=options,
        )
        if filename:
            self.current_file_path = filename
            if self.data_thread:
                self.data_thread.change_file(self.current_file_path)

    def duration_changed(self, value):
        self.duration = value * 60  # Update the duration in seconds

    def recording_stopped(self):
        self.text_edit.append("Recording stopped.")
        self.stop_recording()  # Ensure recording stops when timer hits 0

    def update_time_remaining(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time_label)
        self.timer.start(1000)

    def update_time_label(self):
        minutes = self.duration // 60
        seconds = self.duration % 60
        self.time_label.setText(f"Time Remaining: {minutes:02d}:{seconds:02d}")
        self.duration -= 1
        if self.duration < 0:
            self.timer.stop()
            self.time_label.setText("Time Remaining: 00:00")
            self.recording_stopped()  # Call recording_stopped when timer hits 0


if __name__ == "__main__":
    model = load(os.path.join(script_dir, "trained_model.joblib"))

    app = QApplication(sys.argv)
    mainWindow = MainWindow(model)
    mainWindow.show()
    sys.exit(app.exec_())
