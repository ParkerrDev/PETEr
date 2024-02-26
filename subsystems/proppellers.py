import RPi.GPIO as GPIO
import time

# Pin Definitions
esc_gpio_pin = 18  # GPIO pin number

# Initialize GPIO for ESC
GPIO.setmode(GPIO.BCM)
GPIO.setup(esc_gpio_pin, GPIO.OUT)

# Initialize PWM signal
pwm = GPIO.PWM(esc_gpio_pin, 50)  # 50Hz frequency

# Initial duty cycle (It's important to calibrate your ESC to understand the range)
pwm.start(0)


def calibrate_esc():
    print("Disconnect the battery and press Enter")
    input()
    pwm.ChangeDutyCycle(10)  # Max throttle
    print(
        "Connect the battery now. You'll hear two beeps. Then wait for another 2 seconds and press Enter"
    )
    input()
    pwm.ChangeDutyCycle(5)  # Min throttle
    time.sleep(2)
    print("ESC is now calibrated. Press Enter to continue...")
    input()
    pwm.ChangeDutyCycle(0)  # Stop


def control_esc(speed):
    """Control the ESC speed.
    Speed parameter should be between the calibrated min and max values (e.g., 5-10)."""
    pwm.ChangeDutyCycle(speed)


try:
    calibrate_esc()

    # Example usage
    while True:
        speed = float(input("Enter speed (5 to 10): "))
        control_esc(speed)

except KeyboardInterrupt:
    print("Program stopped")

finally:
    pwm.stop()
    GPIO.cleanup()
