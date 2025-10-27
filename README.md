# 🤖 AI Project Collection

Welcome to a showcase of **AI applications** — ranging from real-time object detection to medical image classification, built with Python, TensorFlow, and cutting-edge computer vision libraries.  

> 🚀 Each project is standalone, modular, and demo-ready — designed to run efficiently on edge devices like **phyBOARD-Pollux i.MX 8M Plus**.

---

## 🧭 Target Platform: phyBOARD-Pollux (i.MX 8M Plus)

This repository is **optimized for deployment** on the NXP-based **phyBOARD-Pollux i.MX 8M Plus** board:

- ✅ Uses TFLite quantized models compatible with **NPU acceleration**
- ✅ Tailored for real-time inference on embedded systems and edge devices
- ✅ Supports NPU acceleration using libvx_delegate.so, with compatibility for TFLite int8 models.

> 🧠 All models have been tested or designed for **real-time edge inference** on the i.MX8MP using **TensorFlow Lite + NPU**.

---

## 🗂️ Project Index

| # | Project | Description |
|--|---------|-------------|
| 01 | [📦 Object Detection](./01-object_detection) | Detects objects in real-time using YOLO or SSD (mobilenet)
| 02 | [🧠 Image Classification](./02-image_classification) | Classifies input images into categories
| 03 | [🎭 Selfie Segmentation](./03-selfie-segmenter) | Removes or replaces selfie backgrounds 
| 04 | [🩺 Pneumonia Detection](./04-pneumonia_detection) | Detects pneumonia from X-rays
| 05 | [🔍 Number Plate Extraction](./05-numberplate_extraction) | Detects and extracts license plates using OCR
| 06 | [🕺 Pose Detection](./06-pose_detection) | Detects human body keypoints
| 07 | [👤 Face Recognition](./07-face_recognition) | Recognizes or verifies faces from images/video 
| 08 | [🖐️ Gesture Detection](./08-gesture_detection) | Detects hand gestures 
| 09 | [🚗 Driver Monitoring System](./09-driver_monitoring_system) | Detects eye-closure and yawning
| 10 | [🛣️ Lane Detection](./10-lane_detection) | Highway road lane detection
| 11 | [🧠 Brain Tumor Detection System](./11-Brain-Tumor-Detection-System) | Brain tumor detection on MRI images

---

## ⚙️ Features

- 📸 Real-time and static image support
- 🧠 Custom + Pretrained Models (TFLite, TensorFlow, PyTorch)
- 🖼️ Built-in visualization for predictions
- 🧩 Optimized for NPU with quantized models (`int8`)
- 🔧 Easy to adapt for edge devices like i.MX8MP , i.MX93 .

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-lightgrey)
![MediaPipe](https://img.shields.io/badge/MediaPipe-AI-green)
![TFLite](https://img.shields.io/badge/TFLite-Quantized-yellow)

- `OpenCV`, `NumPy`, `Matplotlib`, `Flask` (optional)
- TFLite + `libvx_delegate.so` for NPU acceleration
- camera input for /dev/videoX (preferable cam support with USB)
- Display using Wayland sink or OpenCV `imshow` fallback

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/adarshkv159/ai-demos.git
cd ai-demos

# (Optional) Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install shared dependencies
  Refer to the README file provided within each project folder for specific instructions.

