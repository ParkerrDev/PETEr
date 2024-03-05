import RPi.GPIO as GPIO
import time

# Use BCM GPIO numbering
GPIO.setmode(GPIO.BCM)

# Pins configuration
pinA = 2  # Corresponds to DigitalPin nr 2 on the Arduino code
pinB = 3  # Corresponds to DigitalPin nr 3 on the Arduino code

# Setup pins
GPIO.setup(pinA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(pinB, GPIO.IN, pull_up_down=GPIO.PUD_UP)

counter = (
    0  # This variable will increase or decrease depending on the rotation of encoder
)
temp = 0


def ai0(channel):
    global counter
    # Check pinB to determine the direction
    if GPIO.input(pinB) == 0:
        counter += 1
    else:
        counter -= 1
    print_counter()


def ai1(channel):
    global counter
    # Check pinA to determine the direction
    if GPIO.input(pinA) == 0:
        counter -= 1
    else:
        counter += 1
    print_counter()


def print_counter():
    global counter, temp
    if counter != temp:
        print(counter)
        temp = counter


# Attach interrupts
GPIO.add_event_detect(pinA, GPIO.RISING, callback=ai0)
GPIO.add_event_detect(pinB, GPIO.RISING, callback=ai1)

try:
    while True:
        # Main loop does nothing; just waiting for interrupts
        time.sleep(1)
except KeyboardInterrupt:
    print("Program stopped")
finally:
    GPIO.cleanup()
