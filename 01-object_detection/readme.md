# 🔍 Real-Time Object Detection with TensorFlow Lite & OpenCV

This project demonstrates **real-time object detection** using a **quantized MobileNet SSD model** (`ssd_mobilenet_v1_quant.tflite`) with [TensorFlow Lite](https://www.tensorflow.org/lite) and [OpenCV](https://opencv.org/). It is designed to be portable across platforms such as Linux, Windows, and embedded boards (e.g., NXP i.MX8M Plus) and supports **delegate acceleration** (e.g., NPU or GPU).

---

## 📁 Project Structure

.
├── detect.py # Main object detection script
├── labels.py # Label mapping (class index to label name)
├── ssd_mobilenet_v1_quant.tflite # TFLite quantized object detection model
├── README.md # Project documentation

yaml
Copy
Edit

---

## 🧠 Model Information

- **Model**: SSD MobileNet V1 (Quantized)
- **Format**: TensorFlow Lite (`.tflite`)
- **Input Shape**: [1, 300, 300, 3]
- **Output Tensors**:
  - Boxes: `[num_boxes, 4]`
  - Classes: `[num_boxes]`
  - Scores: `[num_boxes]`
  - Number of detections: `[1]`

This model is optimized for performance on edge devices and compatible with TFLite delegates like the **NPU on i.MX8MP** via `libvx_delegate.so`.

---


✅ Dependencies
Install the required Python packages:

bash
Copy
Edit
pip install opencv-python pillow numpy tflite-runtime
Requirements:

Python 3.6+

OpenCV – for video processing and visualization

NumPy – for numerical operations

Pillow – for image utilities

tflite-runtime – for running the TensorFlow Lite model

🚀 Running the Code
1. Run detection using default camera (index 1)
bash
Copy
Edit
python detect.py
2. Run detection using another camera or a video file
bash
Copy
Edit
# Use camera index 0
python detect.py -i 0

# Use a video file
python detect.py -i path/to/video.mp4
3. Run detection with hardware acceleration (NPU/GPU)
bash
Copy
Edit
python detect.py -d libvx_delegate.so
✅ Make sure the specified delegate library (libvx_delegate.so) is available on your device.

📝 Label Mapping (labels.py)
This file should contain a Python dictionary named label2string that maps class indices to human-readable labels:

python
Copy
Edit
label2string = {
    0: 'person',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    # Add more classes as needed
}
Ensure the label indices match those used by your .tflite model.

🎯 Output
The output includes:

🟥 Bounding boxes

🏷️ Class labels

📈 FPS (frames per second)

⏱️ Inference time per frame

Console Output Example
text
Copy
Edit
Detection: (84, 130)-(210, 310) Label: person
FPS: 26  Inference: 15ms
Display
A window will show the live annotated video stream.

Press q to exit the stream.

⚙️ Internal Processing Flow
Initialize video source (camera or file)

Load the TFLite model (optionally with delegate)

Preprocess frame (resize to model input size)

Run inference

Postprocess results:

Draw bounding boxes

Map class indices to labels

Display FPS and inference time

Repeat until user exits

💡 Tips
Use a quantized model (uint8) for best compatibility with hardware acceleration (e.g., NPU).

On NXP i.MX8MP or similar devices, use the libvx_delegate.so delegate.

Performance depends on:

Input size

Camera resolution

Model complexity

Delegate (NPU/GPU/CPU)

✅ Copy This into Your README.md
Let me know if you'd like me to paste the entire updated README as a final clean version!








