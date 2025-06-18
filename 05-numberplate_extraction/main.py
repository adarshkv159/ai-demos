import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading
import time
import re
import os

# === Load OCR TFLite Model ===
try:
    ocr_interpreter = tflite.Interpreter(model_path="license_plate_character_recognition.tflite",
                                         experimental_delegates=[tflite.load_delegate('libvx_delegate.so')])
    print("OCR model loaded on NPU.")
except:
    ocr_interpreter = tflite.Interpreter(model_path="license_plate_character_recognition.tflite")
    print("OCR on CPU.")

ocr_interpreter.allocate_tensors()
ocr_input = ocr_interpreter.get_input_details()
ocr_output = ocr_interpreter.get_output_details()

# === Character Map ===
char_map = {i: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

def predict_character(img):
    img = img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img, axis=0)
    ocr_interpreter.set_tensor(ocr_input[0]['index'], input_tensor)
    ocr_interpreter.invoke()
    output = ocr_interpreter.get_tensor(ocr_output[0]['index'])
    return np.argmax(output)

def recognize_plate(chars):
    text = ''
    for ch in chars:
        ch_rgb = cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)
        resized = cv2.resize(ch_rgb, (28, 28))
        label = predict_character(resized)
        text += char_map[label]
    return text

def segment_characters(plate_img):
    plate_img = cv2.resize(plate_img, (333, 75))
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 15 < h < 70 and 5 < w < 60:
            char = thresh[y:y+h, x:x+w]
            padded = np.full((28, 28), 0, dtype=np.uint8)
            char_resized = cv2.resize(char, (20, 20))
            padded[4:24, 4:24] = char_resized
            char_regions.append((x, padded))
    char_regions = sorted(char_regions, key=lambda tup: tup[0])
    return [img for _, img in char_regions]

# === Detection + OCR Threaded Processing ===
class VideoProcessor:
    def __init__(self, model_path, label_path, video_path):
        self.cap = cv2.VideoCapture(video_path)
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        try:
            self.interpreter = tflite.Interpreter(model_path=model_path,
                                                  experimental_delegates=[tflite.load_delegate('libvx_delegate.so')])
            print("Detection model loaded on NPU.")
        except:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            print("Detection on CPU.")

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        self.last_text = ""
        self.last_text_printed = ""
        self.spoken_plates = set()

        self.last_stable_text = ""
        self.last_stable_time = time.time()

        self.tts_lock = threading.Lock()

    def speak_plate(self, text):
        def speak():
            with self.tts_lock:
                audio_path = "/tmp/plate.wav"
                os.system(f'espeak -w {audio_path} \"{text}\"')
                # Try default audio device first
                ret = os.system(f'aplay -D plughw:0,0 {audio_path} > /dev/null 2>&1')
        threading.Thread(target=speak).start()

    def process(self):
        print("Processing video... Press 'q' to stop.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            if self.float_input:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = input_data.astype(np.uint8)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            for i in range(len(scores)):
                if scores[i] > 0.5:
                    ymin = int(max(1, boxes[i][0] * imH))
                    xmin = int(max(1, boxes[i][1] * imW))
                    ymax = int(min(imH, boxes[i][2] * imH))
                    xmax = int(min(imW, boxes[i][3] * imW))

                    plate_crop = frame[ymin:ymax, xmin:xmax]
                    chars = segment_characters(plate_crop)
                    plate_text = recognize_plate(chars) if chars else ""

                    if re.fullmatch(r'[A-Z]{1}\d{3}[A-Z]{2}', plate_text) or \
                       re.fullmatch(r'\d{2}[A-Z]{2}\d{2}', plate_text) or \
                       re.fullmatch(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', plate_text):

                        current_time = time.time()
                        if plate_text == self.last_stable_text:
                            if (current_time - self.last_stable_time) >= 0.5 and plate_text not in self.spoken_plates:
                                print(f"Detected plate : {plate_text}")
                                self.spoken_plates.add(plate_text)
                                self.speak_plate(plate_text)
                        else:
                            self.last_stable_text = plate_text
                            self.last_stable_time = current_time

                        if plate_text != self.last_text_printed:
                            self.last_text_printed = plate_text
                            self.last_text = plate_text

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = f"{self.labels[int(classes[i])]}: {int(scores[i]*100)}% {plate_text}"
                    cv2.putText(frame, label, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if self.last_text:
                text_size, _ = cv2.getTextSize(self.last_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                text_w, text_h = text_size
                cv2.rectangle(frame, (imW - text_w - 20, 10), (imW - 10, 10 + text_h + 10), (0, 0, 0), -1)
                cv2.putText(frame, self.last_text, (imW - text_w - 15, 10 + text_h),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Detection + OCR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "quant_model_NPU_3k.tflite"
    label_path = "labelmap.txt"
    video_path = "demo.webm"
    processor = VideoProcessor(model_path, label_path, video_path)
    processor.process()

