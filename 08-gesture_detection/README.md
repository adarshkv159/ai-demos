# 🔍Hand Tracking with TFLite (INT8)
 
A lightweight real-time hand tracking application using TFLite integer-quantized models and OpenCV.

---

![Demo GIF](output.gif)

---
 
## 📋 Table of Contents
 
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)
 
  * [Video Mode](#video-mode)
* [Models](#models)
 
---
 
## ✅ Prerequisites
 
* Python 3.6+
* OpenCV
* NumPy
* tflite-runtime
 
You can install required packages via:
 
```bash
pip install opencv-python numpy tflite-runtime
```
 
 
 
> 🔧 If using NPU acceleration, ensure the delegate `.so` file (e.g., `libvx_delegate.so`) is accessible.
 
---
 
## ⚙️ Installation
 
Clone the repository and navigate into the project folder
 
Ensure all models and `anchors.csv` are in place as per the structure below.
 
---
 
## 🗂 Project Structure
 
```
.
├── anchors.csv                                        # Anchor points for palm detection
├── hand_landmark_3d_256_integer_quant.tflite          # Hand landmark model
├── palm_detection_builtin_256_integer_quant.tflite    # Palm detection model
├── hand_tracker.py                                    # Tracker logic and TFLite inference code
├── main.py                                            # Main application script
```
 
---
 
## ▶️ Usage
 
### 📷 Video Mode (Default)
 
To run real-time hand tracking using a webcam or video device:
 
```bash
python3 main.py -i /dev/video0
```
 
### 🕶️ Run on NPU
 
To specify a delegate for hardware acceleration:
 
```bash
python3 main.py -i /dev/video0 -d /usr/lib/libvx_delegate.so
```
 
---
 
## 🧠 Models
 
* **Palm Detection Model**:
  `palm_detection_builtin_256_integer_quant.tflite`
  Detects bounding boxes of palms using anchor-based detection.
 
* **Hand Landmark Model**:
  `hand_landmark_3d_256_integer_quant.tflite`
  Predicts 21 3D hand keypoints from cropped palm regions.
 
* **Anchors File**:
  `anchors.csv`
  Precomputed anchor boxes used by the palm detection model.
 
---
 
> 📝 Press `q` to quit the real-time video window.
> 💡 You can replace the models with updated ones, but ensure the quantization and input shape match.
 
---
 
