import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QThread, pyqtSignal
import serial
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Assuming these paths are correctly set to where your models and data are located
script_dir = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_FILE_RF = f"{script_dir}/trained_model.joblib"
TRAINED_MODEL_FILE_NN = f"{script_dir}/trained_nn_model"


# Function to load models (assumes RandomForest and Neural Network models are saved)
def load_models():
    rf_model = joblib.load(TRAINED_MODEL_FILE_RF)
    nn_model = tf.keras.models.load_model(TRAINED_MODEL_FILE_NN)
    return rf_model, nn_model


class DataThread(QThread):
    data_signal = pyqtSignal(str)

    def __init__(self, arduino, rf_model, nn_model, scaler):
        super().__init__()
        self.arduino = arduino
        self.rf_model = rf_model
        self.nn_model = nn_model
        self.scaler = scaler

    def run(self):
        while True:
            if self.arduino.inWaiting() > 0:
                data = self.arduino.readline()
                decoded_data = data.decode("utf-8").strip()
                # Process and predict
                self.process_data(decoded_data)

    def process_data(self, decoded_data):
        try:
            data_dict = json.loads(decoded_data)
            data_df = pd.DataFrame([data_dict])
            X = self.scaler.transform(data_df)  # Apply scaling

            # Make predictions using both models
            rf_proba = self.rf_model.predict_proba(X)[0][1]
            nn_proba = self.nn_model.predict(X)[0][0]

            # Combine and emit the result
            combined_proba = (rf_proba + nn_proba) / 2
            self.data_signal.emit(
                f"Combined Prediction Probability: {combined_proba:.4f}"
            )
        except Exception as e:
            print(f"Error processing data: {e}")


class MainWindow(QMainWindow):
    def __init__(self, rf_model, nn_model, scaler):
        super().__init__()
        self.setWindowTitle("Live Predictions")
        self.setGeometry(100, 100, 600, 400)

        # Setup UI components
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_recording)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Setup serial port and data thread
        self.arduino = serial.Serial("/dev/ttyACM0", 9600)  # Adjust as needed
        self.data_thread = DataThread(self.arduino, rf_model, nn_model, scaler)
        self.data_thread.data_signal.connect(self.update_text)

    def start_recording(self):
        if not self.data_thread.isRunning():
            self.data_thread.start()

    def stop_recording(self):
        if self.data_thread.isRunning():
            self.data_thread.terminate()

    def update_text(self, text):
        self.text_edit.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load models
    rf_model, nn_model = load_models()

    # Assume scaler is prepared (in real use, fit to your training data)
    scaler = StandardScaler()

    mainWindow = MainWindow(rf_model, nn_model, scaler)
    mainWindow.show()
    sys.exit(app.exec_())
