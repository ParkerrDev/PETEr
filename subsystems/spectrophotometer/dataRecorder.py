# To connect to arduino:

# arduino-cli compile --fqbn arduino:avr:mega ~/Dev/GitHub/SF-Robot/arduino/SparkFun_AS7265x_Arduino_Library/examples/Example1_BasicReadings/Example1_BasicReadings.ino
# arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:mega ~/Dev/GitHub/SF-Robot/arduino/SparkFun_AS7265x_Arduino_Library/examples/Example1_BasicReadings/Example1_BasicReadings.ino
# sudo chmod a+rw /dev/ttyACM0
# sudo usermod -a -G dialout $USER

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import serial
import json


class DataThread(QThread):
    data_signal = pyqtSignal(str)  # Signal to emit decoded data

    def __init__(self, arduino, filename):
        super().__init__()
        self.arduino = arduino
        self.filename = filename
        self.is_running = False

    def run(self):
        while self.is_running:
            data = self.arduino.readline()
            if data:
                try:
                    decoded_data = data.decode(
                        "utf-8"
                    ).strip()  # Decode and strip newline
                except Exception as e:
                    print(e)
                self.data_signal.emit(
                    decoded_data
                )  # Emit the decoded data for UI and file writing

    def start_recording(self, filename):
        self.filename = filename
        self.is_running = True
        self.start()

    def stop_recording(self):
        self.is_running = False
        self.wait()  # Wait for the thread to safely finish


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AS7265x Spectrophotometer Data Recorder")
        self.setGeometry(100, 100, 600, 400)

        self.is_recording = False

        # Initialize serial connection
        self.arduino = serial.Serial(
            "/dev/ttyACM0", 115200
        )  # Adjust the port as necessary

        self.current_filename = ""
        self.data_thread = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        # Set File Location Button
        self.file_location_button = QPushButton("Set File Location")
        self.file_location_button.clicked.connect(self.set_file_location)
        self.layout.addWidget(self.file_location_button)

        # Displayed Data
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

        # Record Button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.record_button)

        # Set layout to central widget
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def set_file_location(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Set File Location",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options,
        )
        if fileName:
            self.current_filename = fileName
            self.file_location_button.setText(f"File Location: {self.current_filename}")

    def toggle_recording(self):
        if self.is_recording:
            if self.data_thread:
                self.data_thread.stop_recording()
                self.data_thread = None
            self.record_button.setText("Start Recording")
        else:
            if self.current_filename:
                self.data_thread = DataThread(self.arduino, self.current_filename)
                self.data_thread.data_signal.connect(self.update_output_and_file)
                self.data_thread.start_recording(self.current_filename)
                self.record_button.setText("Stop Recording")
        self.is_recording = not self.is_recording

    def update_output_and_file(self, data):
        self.output_text.append(data)  # Update UI with new data
        with open(self.current_filename, "a") as file:  # Open file in append mode
            file.write(data + "\n")  # Write data to file


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
