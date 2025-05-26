
🔍 Real-Time Object Detection with TFLite + OpenCV
This project uses TensorFlow Lite, OpenCV, and optional NPU acceleration to perform real-time object detection using a quantized MobileNet SSD model.

📦 Requirements
Install dependencies:

bash
Copy
Edit
pip install opencv-python pillow numpy tflite-runtime
✅ Python 3.6+ required

Libraries Used:

🖼️ opencv-python – Video stream handling & display

🧮 numpy – Numerical operations

🖌️ pillow – Image utilities

🧠 tflite-runtime – Efficient TFLite model inference

🚀 How to Run
🖥️ 1. Default Camera (Index 1)
bash
Copy
Edit
python detect.py
🎥 2. Custom Camera or Video File
bash
Copy
Edit
# Camera index 0
python detect.py -i 0

# Video file
python detect.py -i path/to/video.mp4
⚡ 3. With Hardware Acceleration (e.g., NPU/GPU)
bash
Copy
Edit
python detect.py -d libvx_delegate.so
✅ Make sure libvx_delegate.so exists on your device.

🧾 Label Mapping (labels.py)
Create a labels.py file with:

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
    # ... add more as needed
}
🔁 Must match the model’s class indices.

🎯 What You'll See
✅ Real-time Output
🟥 Bounding Boxes

🏷️ Class Labels

📊 FPS and Inference Time

💬 Console Example
text
Copy
Edit
Detection: (84, 130)-(210, 310) Label: person
FPS: 26  Inference: 15ms
🪟 Display
A window shows the video feed with object detections.

Press q to exit.

🔁 Under the Hood
Open video source (camera or file)

Load .tflite model (optionally with delegate)

Resize frame to model input size

Run inference

Parse output:

Get bounding boxes & labels

Draw overlays

Show FPS & timing

Loop till you quit

💡 Tips & Tricks
Use quantized TFLite models (uint8) for NPU/GPU compatibility.

For NXP i.MX8MP, use libvx_delegate.so to run inference on the NPU.

Improve FPS by:

Reducing video resolution

Choosing lighter models

Using efficient delegates

Let me know if you'd like this saved as a file or turned into a GitHub Gist!
