import sys
import json
import pandas as pd
import os
import serial
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
from joblib import load

script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = f"{script_dir}/trained_model.joblib"

default_directory = "/home/parker/Dev/GitHub/PETEr/subsystems/spectrophotometer/spectralDataClassifier/saved_testing_data"
default_filename = "output.txt"
default_filepath = os.path.join(default_directory, default_filename)

if not os.path.exists(default_directory):
    os.makedirs(default_directory)


class DataThread(threading.Thread):
    def __init__(self, arduino, filename, model, duration, callback):
        super().__init__()
        self.arduino = arduino
        self.filename = filename
        self.model = model
        self.duration = duration
        self.is_running = True
        self.elapsed_time = 0
        self.callback = callback
        self.file = None
        self.open_file()

    def open_file(self):
        if self.file:
            self.file.close()
        self.file = open(self.filename, "a")

    def run(self):
        while self.is_running and self.elapsed_time < self.duration:
            data = self.arduino.readline()
            if data:
                decoded_data = data.decode("utf-8", errors="ignore").strip()
                self.process_data(decoded_data)
                self.elapsed_time += 1
        if self.file:
            self.file.close()

    def stop(self):
        self.is_running = False

    def process_data(self, decoded_data):
        try:
            if "{" in decoded_data and "}" in decoded_data:
                data_dict = json.loads(decoded_data)
                if len(data_dict) == 18:
                    prediction = self.model.predict_proba(pd.DataFrame([data_dict]))[0][1]
                    result = f"Prediction: {prediction*100:.3f}%"
                    self.callback(result)
                    self.file.write(f"Sample: {json.dumps(data_dict)} - {result}\n")
        except Exception as e:
            print(f"Error processing data: {e}")


class MainWindow(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.current_file_path = default_filepath
        self.duration = 2
        self.title("Live Predictions")
        self.geometry("800x600")
        self.data_thread = None
        self.arduino = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
        self.create_widgets()

    def create_widgets(self):
        self.text_edit = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=80, height=20)
        self.text_edit.pack()

        self.start_button = tk.Button(self, text="Start", command=self.start_recording)
        self.start_button.pack()

        self.stop_button = tk.Button(self, text="Stop", command=self.stop_recording)
        self.stop_button.pack()

        self.change_save_location_button = tk.Button(self, text="Change Save Location",
                                                     command=self.change_save_location)
        self.change_save_location_button.pack()

        tk.Label(self, text="Recording Duration (Minutes):").pack()
        self.duration_spinbox = tk.Spinbox(self, from_=1, to=10, command=self.duration_changed)
        self.duration_spinbox.delete(0, "end")
        self.duration_spinbox.insert(0, str(self.duration))
        self.duration_spinbox.pack()

        self.time_label = tk.Label(self, text="Time Remaining: --")
        self.time_label.pack()

    def start_recording(self):
        if not self.data_thread or not self.data_thread.is_alive():
            self.duration = int(self.duration_spinbox.get()) * 60
            self.data_thread = DataThread(
                self.arduino,
                self.current_file_path,
                self.model,
                self.duration,
                self.update_text
            )
            self.data_thread.start()
            self.update_time_remaining()

    def stop_recording(self):
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.stop()

    def change_save_location(self):
        filename = filedialog.asksaveasfilename(initialfile=self.current_file_path)
        if filename:
            self.current_file_path = filename

    def duration_changed(self):
        self.duration = int(self.duration_spinbox.get()) * 60

    def update_text(self, text):
        self.text_edit.insert(tk.END, text + "\n")
        self.text_edit.see(tk.END)

    def update_time_remaining(self):
        minutes = self.duration // 60
        seconds = self.duration % 60
        self.time_label.config(text=f"Time Remaining: {minutes:02d}:{seconds:02d}")
        self.duration -= 1
        if self.duration < 0:
            self.time_label.config(text="Time Remaining: 00:00")
            self.stop_recording()
            self.text_edit.insert(tk.END, "Recording stopped.\n")
            return
        self.after(1000, self.update_time_remaining)


if __name__ == "__main__":
    model = load(os.path.join(script_dir, "trained_model.joblib"))
    app = MainWindow(model)
    app.mainloop()
