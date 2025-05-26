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

## 🔧 Requirements

Make sure the following dependencies are installed:

```bash
pip install opencv-python pillow numpy tflite-runtime
Python 3.6+

OpenCV (for video processing and visualization)

NumPy (for numerical operations)

Pillow (image utilities)

tflite-runtime (for model inference)

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
python detect.py -i 0
bash
Copy
Edit
python detect.py -i path/to/video.mp4
3. Run detection with hardware acceleration (NPU/GPU)
bash
Copy
Edit
python detect.py -d libvx_delegate.so
✅ Make sure the specified delegate library is available on your device.

📝 Label Mapping (labels.py)
This file should contain a Python dictionary named label2string that maps class indices to human-readable labels. Example:

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
    ...
}
Ensure it matches the labels used by the TFLite model.

🎯 Output
Detected objects are displayed with:

Bounding boxes

Class label

FPS (frames per second)

Inference time per frame

Sample console output:

yaml
Copy
Edit
Detection: (84, 130)-(210, 310) Label: person
FPS: 26  Inference: 15ms
Sample display:

A window will show the live video feed with detections annotated.

Press q to exit the video stream.

⚙️ Internal Processing Flow
Initialize video source (camera or file)

Load TFLite model (optionally with delegate)

Preprocess frame: Resize to model input size

Run inference

Postprocess results:

Draw bounding boxes

Map class indices to labels

Display FPS and inference time

Repeat until user exits

💡 Tips
Use a quantized model (uint8) for best delegate/NPU compatibility.

For NXP i.MX8MP or similar platforms, use libvx_delegate.so.

Performance may vary depending on resolution, input size, and hardware acceleration.

