import cv2

import time

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
 
# For int8 quantized model

model_path = "models/model_full_integer_quant.tflite"

model_type = ModelType.TUSIMPLE
 
# Initialize lane detection model

lane_detector = UltrafastLaneDetector(model_path, model_type, use_npu=True, model_dtype='int8')
 
# Initialize webcam

cap = cv2.VideoCapture(0)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)
 
# For FPS calculation

prev_time = time.time()
 
while True:

    ret, frame = cap.read()

    if not ret:

        break
 
    # Detect the lanes

    output_img = lane_detector.detect_lanes(frame)
 
    # Calculate FPS

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time
 
    # Put FPS text on the output image

    cv2.putText(output_img, f"FPS: {fps:.2f}", (20, 30),

                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Show the result

    cv2.imshow("Detected lanes", output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
 
cap.release()

cv2.destroyAllWindows()

 
