import time
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

from gpiozero import LED, DistanceSensor

# -------------------------------
# Ultrasonic (gpiozero)
# -------------------------------
sensor = DistanceSensor(echo=5, trigger=6)

# -------------------------------
# LEDs
# -------------------------------
led_wet = LED(17)
led_dry = LED(27)

led_wet.off()
led_dry.off()

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "/home/piyush/ids_env/New Folder/New folder/model.tflite"
LABEL_PATH = "/home/piyush/ids_env/New Folder/New folder/labels.txt"
IMG_SIZE = 224

# -------------------------------
# Load labels
# -------------------------------
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# -------------------------------
# Load model
# -------------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Camera
# -------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(2)

# -------------------------------
# Timing
# -------------------------------
PREDICTION_INTERVAL = 4
LED_DURATION = 2

cycle_start_time = time.time()

last_label = ""
last_confidence = 0

try:
    while True:
        # à¤¦à¥‚à¤°à¥€ (meters â†’ cm)
        distance = sensor.distance * 100
        print(f"Distance: {distance:.2f} cm")

        # ---------------------------------
        # ONLY run detection if object < 12 cm
        # ---------------------------------
        if distance <= 15:
            current_time = time.time()
            elapsed = current_time - cycle_start_time

            frame = picam2.capture_array()

            # Run prediction every 4 sec
            if elapsed >= PREDICTION_INTERVAL:
                cycle_start_time = current_time
                elapsed = 0

                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                img = np.expand_dims(img, axis=0).astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                pred_index = np.argmax(output)
                prediction = labels[pred_index]
                confidence = float(np.max(output))

                last_label = "Wet" if "Wet" in prediction else "Dry"
                last_confidence = confidence

                print(f"Prediction: {last_label} ({confidence:.2f})")

            # LED control (first 2 sec)
            if elapsed < LED_DURATION:
                if last_label == "Wet":
                    led_wet.on()
                    led_dry.off()
                elif last_label == "Dry":
                    led_wet.off()
                    led_dry.on()
            else:
                led_wet.off()
                led_dry.off()

            # Display
            if last_label != "":
                cv2.putText(frame,
                            f"{last_label} ({last_confidence:.2f})",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            cv2.imshow("Camera", frame)

        else:
            # No object nearby
            led_wet.off()
            led_dry.off()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped")

finally:
    led_wet.off()
    led_dry.off()
    cv2.destroyAllWindows()
    picam2.stop()
