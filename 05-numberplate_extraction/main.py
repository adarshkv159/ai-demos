import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pytesseract
import re
import threading
import queue
import time
import os
 
# TTS function using espeak and aplay
def speak_plate(text):
    audio_file = "/tmp/plate.wav"
    os.system(f'espeak -w {audio_file} "{text}"')
    os.system(f'aplay -D plughw:5,0 {audio_file}')
 
def ocr_worker(ocr_queue, stop_event):
    detected_plates = set()
    last_plate = "No plate detected yet"
    valid_plate_pattern = r'^([A-Z]-\d{3}-[A-Z]{2}|\d{2}-[A-Z]{2}-\d{2}|[A-Z]{2}\d{2}[A-Z]{2}\d{4})$'
 
    while not stop_event.is_set() or not ocr_queue.empty():
        try:
            rois, coords = ocr_queue.get(timeout=0.5)
        except queue.Empty:
            continue
 
        for i, roi in enumerate(rois):
            h, w = roi.shape[:2]
            if h == 0 or w == 0:
                continue
 
            new_w = 120
            new_h = int(h * (new_w / w))
            small_roi = cv2.resize(roi, (new_w, new_h))
 
            gray = cv2.cvtColor(small_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
            text = pytesseract.image_to_string(thresh, config='--psm 7')
            cleaned = text.strip().replace(" ", "").replace("\n", "").upper()
            cleaned = cleaned.replace('O', '0').replace('I', '1').replace('Z', '2')
 
            if re.fullmatch(valid_plate_pattern, cleaned):
                if cleaned not in detected_plates:
                    detected_plates.add(cleaned)
                    print(f"[OCR] Plate: {cleaned}")
                    speak_plate(cleaned)
                    last_plate = cleaned
 
        ocr_worker.last_plate = last_plate
 
def main_detection_loop(video_path, modelpath, lblpath, min_conf=0.5):
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
 
    delegate_path = "/usr/lib/libvx_delegate.so"
    if not os.path.exists(delegate_path):
        print("[ERROR] NPU delegate not found. Falling back to CPU.")
        interpreter = tflite.Interpreter(model_path=modelpath)
    else:
        print("[INFO] Loading NPU delegate from:", delegate_path)
        delegates = [tflite.load_delegate(delegate_path)]
        interpreter = tflite.Interpreter(model_path=modelpath, experimental_delegates=delegates)
 
    interpreter.allocate_tensors()
 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean, input_std = 127.5, 127.5
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
 
    imH, imW = None, None
    ocr_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()
 
    ocr_worker.last_plate = "No plate detected yet"
    ocr_thread = threading.Thread(target=ocr_worker, args=(ocr_queue, stop_event))
    ocr_thread.start()
 
    frame_count = 0
    OCR_EVERY_N_FRAMES = 1
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if imH is None or imW is None:
            imH, imW, _ = frame.shape
 
        frame_count += 1
 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (width, height))
        input_data = np.expand_dims(img_resized, axis=0)
 
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std
 
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
 
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
 
        detections = []
        rois_for_ocr = []
        coords_for_ocr = []
 
        for i in range(len(scores)):
            if scores[i] > min_conf:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))
 
                object_name = labels[int(classes[i])]
                label = f"{object_name}: {int(scores[i] * 100)}%"
 
                detections.append((xmin, ymin, xmax, ymax, label))
                roi = frame[ymin:ymax, xmin:xmax]
                if roi.size > 0:
                    rois_for_ocr.append(roi)
                    coords_for_ocr.append((xmin, ymin, xmax, ymax))
 
        if frame_count % OCR_EVERY_N_FRAMES == 0 and rois_for_ocr:
            try:
                if ocr_queue.full():
                    try:
                        ocr_queue.get_nowait()
                    except queue.Empty:
                        pass
                ocr_queue.put_nowait((rois_for_ocr, coords_for_ocr))
            except queue.Full:
                pass
 
        for (xmin, ymin, xmax, ymax, label) in detections:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 2)
 
        plate_text = ocr_worker.last_plate
        plate_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        top_right_x = imW - plate_size[0] - 20
        top_right_y = 40
 
        cv2.rectangle(frame,
                      (top_right_x - 10, top_right_y - 30),
                      (top_right_x + plate_size[0] + 10, top_right_y + 10),
                      (0, 0, 0), -1)
        cv2.putText(frame, plate_text, (top_right_x, top_right_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
 
        cv2.imshow('Detection + OCR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting detection loop...")
            break
 
    stop_event.set()
    ocr_thread.join()
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main_detection_loop('demo.mp4', 'quant_model_NPU_3k.tflite', 'labelmap.txt')
