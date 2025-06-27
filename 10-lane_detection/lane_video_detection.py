import cv2
import time
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# For int8 quantized model
model_path = "models/model_full_integer_quant.tflite"
model_type = ModelType.TUSIMPLE

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_npu=True, model_dtype='int8')

# Initialize video input
cap = cv2.VideoCapture("input.mp4")

# Create a resizable window
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes
    output_img = lane_detector.detect_lanes(frame)

    # Resize output to 640x480
    output_resized = cv2.resize(output_img, (640, 480))

    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)

    # Add FPS text to the frame
    cv2.putText(output_resized, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Detected lanes", output_resized)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

