import time

import numpy as np

import cv2

import argparse

import tflite_runtime.interpreter as tflite
 
def load_labels(filename):

    with open(filename, 'r') as f:

        return [line.strip() for line in f.readlines()]
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_file', default='mobilenet_v1_1.0_224_quant.tflite', help='.tflite model to be executed')

    parser.add_argument('-l', '--label_file', default='labels.txt', help='name of file containing labels')

    parser.add_argument('-d', '--delegate', default='', help='delegate path')

    parser.add_argument('--input_mean', default=127.5, type=float, help='input mean')

    parser.add_argument('--input_std', default=127.5, type=float, help='input std deviation')

    parser.add_argument('--num_threads', default=None, type=int, help='number of threads')

    parser.add_argument('--camera_id', default=0, type=int, help='Camera device ID (/dev/video1 means 1)')

    parser.add_argument('--threshold', default=0.6, type=float, help='Score threshold for displaying results')

    args = parser.parse_args()
 
    # Load model

    if args.delegate:

        ext_delegate = [tflite.load_delegate(args.delegate)]

        interpreter = tflite.Interpreter(model_path=args.model_file, experimental_delegates=ext_delegate)

    else:

        interpreter = tflite.Interpreter(model_path=args.model_file)
 
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()
 
    floating_model = input_details[0]['dtype'] == np.float32

    height = input_details[0]['shape'][1]

    width = input_details[0]['shape'][2]
 
    labels = load_labels(args.label_file)
 
    # Open the video stream

    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():

        print("Cannot open camera")

        exit()
 
    while True:

        ret, frame = cap.read()

        if not ret:

            print("Failed to grab frame")

            break
 
        # Preprocess the frame

        input_data = cv2.resize(frame, (width, height))

        input_data = np.expand_dims(input_data, axis=0)
 
        if floating_model:

            input_data = (np.float32(input_data) - args.input_mean) / args.input_std
 
        interpreter.set_tensor(input_details[0]['index'], input_data)
 
        start_time = time.time()

        interpreter.invoke()

        stop_time = time.time()
 
        output_data = interpreter.get_tensor(output_details[0]['index'])

        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
 
        y_pos = 30

        for i in top_k:

            score = float(results[i] / 255.0) if not floating_model else float(results[i])

            if score < args.threshold:

                continue

            label = labels[i]

            print('{:08.6f}: {}'.format(score, label))

            cv2.putText(frame, f'{label}: {score:.2f}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            y_pos += 30
 
        # Display inference time

        inf_time = (stop_time - start_time) * 1000

        cv2.putText(frame, 'Inference: {:.2f} ms'.format(inf_time), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
        # Show frame

        cv2.imshow('TFLite Classification', frame)
 
        # Press 'q' to quit

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
 
    cap.release()

    cv2.destroyAllWindows()

 