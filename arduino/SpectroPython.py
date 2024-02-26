import serial
import json

arduino = serial.Serial('/dev/ttyACM0', 115200)  # Replace '/dev/ttyACM0' with the port where your Arduino is connected

while True:
    data = arduino.readline()
    if data:
        decoded_data = data.decode('utf-8')
        if '{' in decoded_data and '}' in decoded_data:
            print(data) 
        try:
            json_data = json.loads(decoded_data)
        except json.JSONDecodeError:
            continue
