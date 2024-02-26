import serial  # pip install pyserial


def get_data():
    while True:
        arduino = serial.Serial("/dev/ttyACM0", 9600)
        data = arduino.readline()
        if data:
            yield data
