import sys
import json
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QFileDialog,
)
from PyQt5.QtCore import QThread, pyqtSignal
import serial
from joblib import load
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = f"{script_dir}/trained_model.joblib"


class DataThread(QThread):
    data_signal = pyqtSignal(str)

    def __init__(self, arduino, filename, model):
        super(DataThread, self).__init__()
        self.arduino = arduino
        self.filename = filename
        self.model = model
        self.is_running = True

    def run(self):
        while self.is_running:
            data = self.arduino.readline()
            if data:
                decoded_data = data.decode("utf-8", errors="ignore").strip()
                self.process_data(decoded_data)

    def process_data(self, decoded_data):
        try:
            # Convert the incoming data to a format suitable for prediction
            data_dict = json.loads(decoded_data)
            data_df = pd.DataFrame([data_dict])
            # Assuming the model expects a DataFrame and outputs probabilities
            prediction = self.model.predict_proba(data_df)[0][
                1
            ]  # Probability of being class '1'
            result = f"Prediction: {prediction*100:.3f}%"
            self.data_signal.emit(result)
        except Exception as e:
            print(f"Error processing data: {e}")

    def stop(self):
        self.is_running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self, model):
        super(MainWindow, self).__init__()
        self.model = model
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AS7265x Live Spectral PETE Predictions")
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

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.arduino = serial.Serial("/dev/ttyACM0", 115200)
        self.data_thread = DataThread(self.arduino, "output.txt", self.model)
        self.data_thread.data_signal.connect(self.update_text)

    def start_recording(self):
        if not self.data_thread.isRunning():
            self.data_thread.start()

    def stop_recording(self):
        self.data_thread.stop()

    def update_text(self, text):
        self.text_edit.append(text)


if __name__ == "__main__":
    model = load(model_filename)  # Load your trained model

    app = QApplication(sys.argv)
    mainWindow = MainWindow(model)
    mainWindow.show()
    sys.exit(app.exec_())
