import numpy as np
import json
import serial

# Predefined spectral signature for PET plastic (example values)
known_pet_signature = np.array(
    [
        4953.84,
        464.83,
        1658.52,
        159.71,
        149.62,
        148.47,
        76.52,
        82.27,
        200.36,
        62.03,
        107.02,
        25.28,
        44.75,
        30.77,
        24.14,
        21.14,
        8.81,
        15.76,
    ]
)

# Threshold for identifying PET plastic
identification_threshold = 0.5


def read_spectral_data_from_serial(serial):
    """Reads spectral data from an already opened serial port."""
    serial_buffer = ""
    while True:
        part = serial.readline().decode("utf-8").strip()
        # print(part)
        serial_buffer += part
        if serial_buffer.startswith("{") and serial_buffer.endswith("}"):
            break
        elif len(serial_buffer) > 0 and not serial_buffer.startswith("{"):
            serial_buffer = ""
    try:
        data_dict = json.loads(serial_buffer)
        data_values = np.array(list(data_dict.values()))
        return data_values
    except json.JSONDecodeError:
        print("Invalid JSON format received:", serial_buffer)
        return None


class SpectralAnalyzer:
    def __init__(self, serial_port):
        self.serial_port = serial_port

    def process_data(self, data):
        try:
            similarity = np.linalg.norm(data - known_pet_signature)
            # Normalize similarity to the range [0.01, 1]
            normalized_similarity = max(0.01, 1 - similarity / 9500)
            return normalized_similarity
        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    def run(self):
        while True:
            data = read_spectral_data_from_serial(self.serial_port)
            if data is not None:
                try:
                    prediction = self.process_data(data)
                    # print(prediction)
                    if prediction is not None:
                        print(prediction)
                        if prediction >= identification_threshold:
                            print("The sample is likely PET plastic.")
                        else:
                            print("The sample is not PET plastic.")
                    else:
                        print("Failed to process data.")
                except Exception as e:
                    print(f"Error processing data: {e}")
            else:
                print("Failed to read valid spectral data.")


if __name__ == "__main__":
    # Open the serial port here
    try:
        with serial.Serial("/dev/ttyACM0", 115200, timeout=1) as ser:
            analyzer = SpectralAnalyzer(ser)
            analyzer.run()
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
