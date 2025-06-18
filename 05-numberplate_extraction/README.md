# 🔍 Number Plate Detection and Extraction 

This project demonstrates **number plate detection and extraction** using a **quantized TFLite model**  with **TensorFlow Lite** and **OpenCV**.

Designed for cross-platform use (Linux, Windows, embedded boards like NXP i.MX8M Plus), it supports **hardware acceleration** via delegates like **NPU or GPU**.

---

![Demo GIF](output.gif)

---

## 📁 Project Structure

```
.
├── main.py                                    # Your main script 
├── quant_model_NPU_3k.tflite                  # Quantized TFLite model 
├── license_plate_character_recognition.tflite # Quantized TFLite model
├── labelmap.txt                               # Label mapping (class index to name)  
├── demo.webm                                  # video used in the script 
├── README.md                                  # This documentation
```

---

## 🧠 Model Information

**Number plate Detection Model**

- **Model**: Number plate Detection Model (Quantized)  
- **Format**: TensorFlow Lite (`.tflite`)

**Number plate Extraction Model**

- **Model**: OCR Model (Quantized)  
- **Format**: TensorFlow Lite (`.tflite`)  

✅ Optimized for edge devices  
🧠 Compatible with NPU delegate (`libvx_delegate.so`) on platforms like i.MX8MP

---

## ✅ Dependencies

Install with:

```bash
apt install espeak alsa-utils
pip install opencv-python tflite-runtime pytesseract
```

### Requirements:
- Python 3.6+
- OpenCV – for video stream processing and display
- TFLite Runtime – for inference
- espeak – command-line Text-to-Speech (TTS) engine
- alsa-utils – ALSA sound utilities including aplay for audio playback

### 🔎 Note  
The `opencv-python` package automatically installs the latest version of **NumPy** that is compatible with your Python version.  
However, this program (or one of its dependencies) requires **NumPy version 1.x**, because modules compiled against NumPy 1.x may crash when used with NumPy 2.x or later.

To fix this issue, downgrade NumPy by running:  
```bash
pip install "numpy<2.0"
```
---

## 🚀 How to Run

### 1️⃣ Using a video file:

```bash
python main.py
```
> ✅ Ensure `libvx_delegate.so` exists on your device.If the delegate .so is missing, script will raise an error and stop.

---

## 📝 Label Mapping (`labelmap.txt`)

This file maps class indices to human-readable labels:

```text
   license
```

> 🔁 Ensure these labels correspond exactly to the classes your .tflite model was trained on, so that predictions map correctly to meaningful names.

---

## 🎯 Output

- 🏷️ Detected license plate region(s) with bounding boxes and confidence scores drawn on video frames
- 🔠 Recognized license plate text overlaid on the video in real time (top-right corner of the frame)
- 🔊 Spoken output of new, valid license plate numbers using espeak + aplay
- 📤 Console log messages showing detected and recognized license plate strings

### 📟 Console Output Example

```text
[OCR] Detected Plate: R183JF
```

### 🖼️ Display

- A window titled "Detection + OCR" shows the video with predicted class label and confidence score
- The latest detected license plate shown at the top-right
- Press **`q`** to quit.

---

## ⚙️ Internal Processing Flow

1. Initialize video source from file (demo.webm) using OpenCV.
2. Load TFLite object detection model (quant_model_NPU_3k.tflite) with optional libvx_delegate for NPU acceleration.
3. Load TFLite OCR model (license_plate_character_recognition.tflite) with optional NPU delegate.
4. Read class labels from labelmap.txt for detected objects.
5. Capture frame from video source.
6. Preprocess frame by converting BGR to RGB and resizing to the model’s expected input shape.
7. Run object detection inference on the preprocessed frame using the detection model.
8. Extract bounding boxes, class indices, and confidence scores from model output and filter based on threshold.
9. Crop license plate regions, segment individual characters using contour-based approach, and classify each character with the OCR model.
10. Validate recognized text using regex patterns for Indian number plates and, if stable, use espeak and aplay to speak the detected plate once.
11. Display annotated frame with bounding boxes, labels, and recognized plate text, and repeat the loop until video ends or 'q' is pressed.

---

## 💡 Tips

- ✅ Use **quantized models (uint8)** for better hardware compatibility
- 🚀 For NXP i.MX8MP, use **`libvx_delegate.so`** to run on the NPU
- 📏 Adjust input size/resolution to balance accuracy and performance
