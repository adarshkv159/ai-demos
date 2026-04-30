# 🎥 Edge-Accelerated Monocular Depth Perception with Real-Time 3D Point Cloud Visualization on Embedded NPU

This project demonstrates a **real-time monocular depth estimation pipeline** using a quantized **MiDaS v2.1 Small TFLite model**, accelerated on embedded hardware (e.g., i.MX8M Plus NPU).

It captures input from a **webcam or image**, performs **INT8 depth inference**, and renders an interactive **3D point cloud visualization in the browser using Three.js**.

The system is designed for **edge AI deployment**, supporting:

* ⚡ NPU acceleration (via VX delegate)
* 🖥 CPU fallback
* 🌐 Web-based 3D visualization + editing tools
* 📦 Export (PLY / JSON / PNG)

---

## 📸 Output

![Demo GIF](output.gif)


---

## 📁 Project Structure

```
├── midas_pointcloud.html        # Frontend (Three.js visualization UI)
├── server.py                   # Backend inference server (TFLite + NPU)
├── midas_v2.1_small_quant_recalib.tflite   # INT8 quantized depth model
```

---

## 🧠 Model Information

* **Model**: MiDaS v2.1 Small
* **Format**: TensorFlow Lite (INT8 quantized)
* **Input Size**: 256 × 256 × 3
* **Output**: 256 × 256 depth map
* **Normalization**:

  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`
* **Quantization**:

  * Input Scale: `0.01865844801068306`
  * Input Zero Point: `-14`
  * Output Scale: `10.557392120361328`
  * Output Zero Point: `-128`

### ⚙️ Runtime Features

* NPU acceleration using **VX delegate**
* Automatic fallback to CPU (multi-threaded)
* Runtime backend switching (`/set_backend` API)

Inference pipeline:

```
RGBA Input → Preprocessing → INT8 Quantization → TFLite Inference
→ Dequantization → Normalization → Depth Map → Point Cloud
```

---

## 📦 Dependencies

Install required Python packages:

```bash
pip install numpy==1.26.4 \
            tflite-runtime==2.15.0 \
            opencv-python==4.9.0 \
            flask==3.1.1
```

---

## ▶️ How to Run

### 1️⃣ Start the Backend Server

```bash
python3 server.py
```

Server will start at:

```
http://0.0.0.0:5001
```

* Automatically loads model
* Tries NPU first → falls back to CPU if unavailable 

---

### 2️⃣ Open the Web UI

Open in browser:

```
http://localhost:5001
```
Remote access in same network for headless:
```
http://<IP_ADD_BOARD>:5001
```

This serves the frontend interface directly from the server 

---

### 3️⃣ Run Inference

#### 📷 Option A: Webcam

* Click **START CAM**
* Real-time depth + point cloud visualization

#### 🖼 Option B: Image

* Upload JPG/PNG
* Generates static editable 3D point cloud

---

### 4️⃣ Features in UI

* 🔁 Switch between **NPU / CPU**
* 🎛 Adjust:

  * Depth scaling
  * Point density
  * Deformations (twist, wave, noise, sphere)
* 📸 Capture frames
* ✏️ Edit frozen point clouds
* 💾 Export:

  * `.ply` → for MeshLab / Blender
  * `.json` → raw data
  * `.png` → rendered view

Frontend built using **Three.js for real-time rendering and interaction** 

---

## 🔌 API Endpoints

| Endpoint       | Method | Description                      |
| -------------- | ------ | -------------------------------- |
| `/status`      | GET    | Check server + backend (NPU/CPU) |
| `/infer`       | POST   | Run depth inference              |
| `/set_backend` | POST   | Switch between NPU and CPU       |

---

## ⚡ Hardware Target

* Designed for **embedded edge AI systems**
* Tested on:

  * i.MX8M Plus (NPU via VX delegate)
* Compatible with:

  * ARM64 (aarch64)
  * Linux-based edge devices

---

## 🧩 Key Highlights

* 🧠 **Efficient INT8 model** for edge deployment
* ⚡ **NPU acceleration support**
* 🌐 **Browser-based 3D visualization**
* 🎮 **Interactive point cloud editing**
* 📦 **Exportable 3D data formats**

