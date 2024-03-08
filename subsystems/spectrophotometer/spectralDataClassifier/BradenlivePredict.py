import sys
import json
import pandas as pd
import numpy as np
import statistics
from statistics import stdev
import random
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
from combinedSpectralDataClassifer import combinedClassifier

pList = []
sList = []
rList = []

script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = f"{script_dir}/trained_model.joblib"


class DataThread(QThread):
    data_signal = pyqtSignal(str)
    global pList
    global sList
    global rList
    global decoded_data
    global result

    def statistics(self, list1):
        global pList
        global sList
        mean = statistics.mean(list1)
        mode = statistics.mode(list1)
        stdev = statistics.stdev(list1)
        minimum = np.quantile(list1, 0)
        Q1 = np.quantile(list1, 0.25)
        median = np.quantile(list1, 0.5)
        Q3 = np.quantile(list1, 0.75)
        maximum = np.quantile(list1, 1)
        IQR = Q3 - Q1
        return f"mean: {mean} \nmode: {mode} \nstandard deviation: {stdev} \nminimum: {minimum} \nQuartile 1: {Q1} \nmedian: {median} \nQuartile 3: {Q3} \nmaximum: {maximum} \nIQR: {IQR}"

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
            global rList
            prediction = combinedClassifier.combined_predict(str(decoded_data))
            result = f"Prediction: {prediction:.4f}"
            rList.append(decoded_data + " ; " + result)
            self.data_signal.emit(result)

            if len(pList) < 301:  # Only append if less than 301 predictions
                pList.append(prediction)
            else:
                self.data_signal.emit("STOPPED RECORDING")
                self.stop()  # Stop collecting data if 301 predictions have been reached
        except Exception as e:
            print(f"Error processing data: {e}")

    def stop(self):
        self.is_running = False
        self.arduino.close()  # Close the serial connection
        self.wait()  # Wait for the thread to finish
        self.write_to_file()  # Write pList and sList to a file

    def write_to_file(self):
        global pList
        global sList
        global rList
        self_dir = os.path.dirname(os.path.abspath(__file__))
        filename = f"{self_dir}/recorded_values.txt"
        sList = random.sample(pList, 30)
        with open(filename, "w") as f:
            for value in rList:
                f.write(f"{value}\n")
            for value in pList:
                f.write(f"{value}\n")
            f.write(self.statistics(pList))
            for value in sList:
                f.write(f"{value}\n")
            f.write(self.statistics(sList))
        print(f"Data saved to {filename}")


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

        self.arduino = serial.Serial("COM3", 115200)
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
