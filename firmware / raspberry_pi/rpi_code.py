import time
import numpy as np
import cv2
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "/home/piyush/ids_env/New Folder/New folder/model.tflite"
LABEL_PATH = "/home/piyush/ids_env/New Folder/New folder/labels.txt"
IMG_SIZE = 224

# -------------------------------
# GPIO Pins
# -------------------------------
LED_WET_PIN = 17
LED_DRY_PIN = 27
TRIG_PIN = 23
ECHO_PIN = 24

# -------------------------------
# Load Labels
# -------------------------------
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Labels:", labels)

# -------------------------------
# Load Model
# -------------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# GPIO Setup
# -------------------------------
GPIO.setmode(GPIO.BCM)

GPIO.setup(LED_WET_PIN, GPIO.OUT)
GPIO.setup(LED_DRY_PIN, GPIO.OUT)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

GPIO.output(LED_WET_PIN, GPIO.LOW)
GPIO.output(LED_DRY_PIN, GPIO.LOW)

# -------------------------------
# Functions
# -------------------------------
def blink_led(pin, duration=2):
    GPIO.output(pin, GPIO.HIGH)
    print(f"LED ON (Pin {pin})")
    time.sleep(duration)
    GPIO.output(pin, GPIO.LOW)
    print(f"LED OFF (Pin {pin})")

def get_distance():
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.05)

    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()

    timeout = time.time() + 0.04  # 40ms timeout

    # Wait for echo start
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
        if start_time > timeout:
            return 999  # No signal

    # Wait for echo end
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
        if stop_time > timeout:
            return 999

    duration = stop_time - start_time
    distance = duration * 17150
    return distance

# -------------------------------
# Camera Setup
# -------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(2)

last_classification = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
try:
    while True:
        dist = get_distance()

        if dist < 18:
            current_time = time.time()

            # Run every 4 seconds
            if current_time - last_classification >= 4:
                last_classification = current_time

                frame = picam2.capture_array()
                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                img = np.expand_dims(img, axis=0).astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]['index'])
                pred_index = np.argmax(output)
                prediction = labels[pred_index]
                confidence = np.max(output)

                label = "Wet" if "Wet" in prediction else "Dry"

                print(f"Prediction: {label} ({confidence:.2f})")

                # LED Control
                if label == "Wet":
                    blink_led(LED_WET_PIN, 2)
                else:
                    blink_led(LED_DRY_PIN, 2)

                # Display
                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                cv2.imshow("Camera", frame)
                cv2.waitKey(1)

        else:
            print(f"No object detected: {dist:.2f} cm")
            time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopped")

finally:
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
