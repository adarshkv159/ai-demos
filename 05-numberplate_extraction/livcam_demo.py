import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import pytesseract
import re
import threading


# Camera capture thread
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# Load interpreter with NPU fallback
def load_tflite_interpreter(model_path):
    try:
        delegate = tflite.load_delegate("libvx_delegate.so")
        print("INFO: NPU delegate loaded successfully.")
        return tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    except Exception as e:
        print(f"WARNING: Failed to load NPU delegate. Falling back to CPU. Error: {e}")
        return tflite.Interpreter(model_path=model_path)


def tflite_detect_camera(modelpath, lblpath, min_conf=0.5):
    # Load labels
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load TFLite model with NPU fallback
    interpreter = load_tflite_interpreter(modelpath)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Start camera thread
    cam = CameraStream()
    time.sleep(1)  # Let it warm up

    total_fps = 0
    total_time = 0
    last_plate = "No Number Plate Detected"

    pattern_10 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    pattern_9 = r'^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$'

    while True:
        loop_start = time.time()
        ret, frame = cam.read()
        if not ret:
            print("Camera disconnected.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                object_name = labels[int(classes[i])]
                label_conf = f"{object_name}: {int(scores[i] * 100)}%"

                plate_img = frame[ymin:ymax, xmin:xmax]
                if plate_img.size == 0:
                    continue

                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                plate_gray = cv2.medianBlur(plate_gray, 3)
                plate_gray = cv2.erode(plate_gray, np.ones((3, 3), np.uint8), iterations=1)

                try:
                    plate_text = pytesseract.image_to_string(
                        plate_gray,
                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )
                    plate_text = ''.join(e for e in plate_text if e.isalnum()).upper()

                    if re.match(pattern_10, plate_text) or re.match(pattern_9, plate_text):
                        display_text = f"{label_conf} | {plate_text}"

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        labelSize, baseLine = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                      (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, display_text, (xmin, label_ymin - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        if plate_text != last_plate:
                            print(f"Valid Plate Detected: {plate_text}")
                            last_plate = plate_text
                    else:
                        continue

                except Exception as e:
                    print(f"OCR Error: {e}")

        # FPS and inference time
        loop_end = time.time()
        total_fps += 1
        total_time += (loop_end - loop_start)
        fps = int(total_fps / total_time) if total_time > 0 else 0
        invoke_time = int((loop_end - loop_start) * 1000)
        msg = f"FPS: {fps} | Time: {invoke_time}ms"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Top-right plate
        plate_text = last_plate
        plate_size, baseLine = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        top_right_x = imW - plate_size[0] - 10
        top_right_y = 30
        cv2.rectangle(frame,
                      (top_right_x - 10, top_right_y - 30),
                      (top_right_x + plate_size[0] + 10, top_right_y + 10),
                      (0, 0, 0), -1)
        color = (0, 255, 255) if plate_text != "No Number Plate Detected" else (0, 0, 255)
        cv2.putText(frame, plate_text, (top_right_x, top_right_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('License Plate Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tflite_detect_camera("quant_model_NPU_3k.tflite", "labelmap.txt")
