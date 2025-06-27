# 🛣️ Lane Detection using TensorFlow Lite INT8 Model
 
A real-time lane detection system using TensorFlow Lite INT8 quantized model for efficient inference with NPU acceleration. This project provides multiple inference modes including image, video, and webcam detection.

---
<p align="center">
  <img src="output.gif" alt="Image Mode Demo" width="45%">
  <img src="output.gif" alt="Video Mode Demo" width="45%">
</p>


---


## 📑 Table of Contents
 
* [🔧 Prerequisites](#-prerequisites)
* [⚙️ Installation](#-installation)
* [📁 Project Structure](#-project-structure)
* [🚀 Usage](#-usage)
 
  * [🖼️ Image Mode](#️-image-mode)
  * [🎞️ Video Mode](#-video-mode)
  * [📷 Webcam Mode](#-webcam-mode)
* [🧠 Models](#-models)
* [🙏 Acknowledgments](#-acknowledgments)
* [📝 License](#-license)
 
## 🔧 Prerequisites
 
Before running this project, ensure you have the following requirements:
 
* Python 3.7 or higher
* OpenCV (cv2)
* NumPy
* TensorFlow Lite runtime
* Webcam (for webcam mode)
 
## ⚙️ Installation
 
1. Clone this repository:
 
```bash
git clone <your-repository-url>
cd lane-detection
```
 
2. Install required dependencies:
  
```bash
pip install tflite-runtime opencv-python numpy
```
 
## 📁 Project Structure
 
```
.
├── imageLaneDetection.py                # Image inference script
├── input.jpg                            # Sample input image
├── input.mp4                            # Sample input video
├── lane_detection_webcam_fps.py         # Webcam inference with FPS counter
├── lane_video_detection.py              # Video inference script
├── models/
│   └── model_full_integer_quant.tflite  # INT8 quantized TFLite model
└── ultrafastLaneDetector/
    ├── __init__.py
    └── ultrafastLaneDetector.py         # Main inference engine
```
 
## 🚀 Usage
 
### 🖼️ Image Mode
 
Process a single image for lane detection:
 
```bash
python imageLaneDetection.py
```
 
This script will:
 
* Load the input image (`input.jpg`)
* Perform lane detection inference
* Display the result with detected lanes highlighted
* Save the output image
 
### 🎞️ Video Mode
 
Process a video file for lane detection:
 
```bash
python lane_video_detection.py
```
 
Features:
 
* Processes video frame by frame
* Displays real-time lane detection results
* Supports various video formats (mp4, avi, etc.)
 
### 📷 Webcam Mode
 
Real-time lane detection using webcam:
 
```bash
python lane_detection_webcam_fps.py
```
 
Features:
 
* Real-time processing from webcam feed
* FPS counter for performance monitoring
* Live visualization of detected lanes
* Press 'q' to quit
 
## 🧠 Models
 
This project uses an INT8 quantized TensorFlow Lite model for efficient inference:
 
* **Model**: `model_full_integer_quant.tflite`
* **Quantization**: Full integer (INT8) quantization
* **Size**: Optimized for mobile and edge devices
* **Performance**: Fast inference with requiring NPU acceleration
 
### Model Details
 
The model is based on Ultra-Fast Lane Detection architecture and has been quantized to INT8 precision for:
 
* Reduced model size
* Faster inference on NPU
* Lower memory consumption
* Maintained accuracy for lane detection tasks
 
 
## 🙏 Acknowledgments
 
This project builds upon excellent work from the community:
 
* **Inference Code**: Inspired and adapted from [TfLite-Ultra-Fast-Lane-Detection-Inference](https://github.com/ibaiGorordo/TfLite-Ultra-Fast-Lane-Detection-Inference/tree/main) by ibaiGorordo
* **TensorFlow Lite Model**: Obtained from [PINTO\_model\_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection) by PINTO0309
 
Special thanks to the original authors for their contributions to the open-source community.
 
## 🛠️ Features
 
* ✅ **Multiple Input Modes**: Support for image, video, and webcam inputs
* ✅ **INT8 Quantization**: Optimized model for efficient inference
* ✅ **Real-time Processing**: Suitable for live applications
* ✅ ** NPU Accelaration**: NPU inference
* ✅ **Cross-platform**: Works on various operating systems
* ✅ **Easy to Use**: Simple command-line interface
