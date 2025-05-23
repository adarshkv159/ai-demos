#ght 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import cv2
import time, os
import argparse

from labels import label2string

MODEL_PATH = "ssd_mobilenet_v1_quant.tflite"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='1',  # Changed from '/dev/video0' to '0' for Windows compatibility
    help='input to be classified (camera index or video file path)')
parser.add_argument(
    '-d',
    '--delegate',
    default='',
    help='delegate path')
args = parser.parse_args()

# Camera initialization (Windows compatible)
if args.input.isdigit():
    cap_input = int(args.input)  # Use camera index if number is provided
else:
    cap_input = args.input  # Otherwise try as file path
    
vid = cv2.VideoCapture(cap_input)
if not vid.isOpened():
    print(f"Error: Could not open video source {args.input}")
    exit(1)

# Initialize TensorFlow Lite interpreter
if args.delegate:
    ext_delegate = [tflite.load_delegate(args.delegate)]
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_delegates=ext_delegate)
else:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Performance tracking
msg = ""
total_fps = 0
total_time = 0

# Main processing loop
ret, frame = vid.read()
if frame is None:
    print(f"Error: Could not read frame from source {args.input}")
    vid.release()
    exit(1)

try:
    while ret:
        total_fps += 1
        loop_start = time.time()

        # Preprocess frame
        img = cv2.resize(frame, (width, height)).astype(np.uint8)
        input_data = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        invoke_start = time.time()
        interpreter.invoke()
        invoke_end = time.time()

        # Get results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        labels = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        number = interpreter.get_tensor(output_details[3]['index'])[0]

        # Draw detections
        for i in range(int(number)):
            if scores[i] > 0.5:  # Confidence threshold
                box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                x0 = max(2, int(box[1] * frame.shape[1]))
                y0 = max(2, int(box[0] * frame.shape[0]))
                x1 = int(box[3] * frame.shape[1])
                y1 = int(box[2] * frame.shape[0])

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(frame, label2string[labels[i]], (x0, y0 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print(f"Detection: ({x0},{y0})-({x1},{y1}) Label: {label2string[labels[i]]}")

        # Calculate and display performance metrics
        loop_end = time.time()
        total_time += (loop_end - loop_start)
        fps = int(total_fps / total_time)
        invoke_time = int((invoke_end - invoke_start) * 1000)
        msg = f"FPS: {fps}  Inference: {invoke_time}ms"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display frame
        cv2.imshow("Object Detection", frame)

        # Read next frame or exit on 'q' key
        ret, frame = vid.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    vid.release()
    cv2.destroyAllWindows()
