# PoseNet Single-Person Inference with NPU Delegate on Phycore-imx8m plus (Python)
 
This project demonstrates how to run google-corel [project - PoseNet](https://github.com/google-coral/project-posenet)  on a phycore-imx8m plus  using the TensorFlow Lite Runtime. It attempts to load an NPU delegate if available, falling back to CPU inference otherwise. After inference, it uses DBSCAN clustering to select and render keypoints belonging to the main (highest‐confidence) person in the frame.
 
---

![Demo Image](output.png)

---
 
## Table of Contents
 
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [How It Works](#how-it-works)  
  - [1. Loading the TFLite Model (with NPU Delegate)](#1-loading-the-tflite-model-with-npu-delegate)  
  - [2. Preprocessing & Inference](#2-preprocessing--inference)  
  - [3. Decoding Heatmaps & Offsets](#3-decoding-heatmaps--offsets)  
  - [4. Selecting the Main Person via DBSCAN](#4-selecting-the-main-person-via-dbscan)  
  - [5. Drawing Keypoints & Skeleton](#5-drawing-keypoints--skeleton)  
- [Dependencies](#dependencies)  
- [Troubleshooting](#troubleshooting)  
- [License](#license)  
 
---
 
## Prerequisites
 
1. **Operating System**  
   - Ubuntu 20.04 or newer (tested on Ubuntu 22.04 with Python 3.11).  
2. **Python Version**  
   - Python 3.8+ (3.10 or 3.11 recommended).  
3. **Camera**  
   - A webcam (USB or built‐in) recognized by OpenCV.  
4. **Hardware Accelerator (Optional)**  
   - An NPU  delegate library (`libvx_delegate.so`) installed in `/usr/lib/` or Facematch BSP of imx8m plsu rootfs.  
 
---
 
## Installation
 
1. **Clone or download this repository**  
   ```bash
   git clone https://YOUR_REPO_URL_HERE.git
   cd YOUR_REPO_DIRECTORY
   ```
 
2. **Create a Python virtual environment (recommended)**
 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
 
3. **Install system packages**
   On Ubuntu:
 
   ```bash
   sudo apt update
   sudo apt install -y \
     build-essential \
     libopencv-dev \
     python3-dev \
     python3-venv \
     pkg-config \
     cmake
   ```
 
4. **Install Python dependencies**
 
   ```bash
   pip install --upgrade pip
   pip install \
     numpy \
     opencv-python-headless \
     tflite-runtime \
     scikit-learn
   ```
 
   * **`tflite-runtime`**: Provides the minimal TFLite interpreter without the full TensorFlow dependency.
   * **`opencv-python-headless`**: If you do not need GUI (imshow) support, otherwise install `opencv-python`.
   * **`scikit-learn`**: Required for DBSCAN clustering and `StandardScaler`.
 
5. **Place the PoseNet model file**
   Ensure the quantized PoseNet model is saved as:
 
   ```text
   models/posenet_mobilenet_v1_075_353_481_quant.tflite
   ```
 
   (You can download a prebuilt TFLite quantized PoseNet from the (https://github.com/google-coral/project-posenet/tree/master/models/mobilenet/components).)
 
6. **verify NPU delegate**
 
   * NPU delegate library (e.g., `libvx_delegate.so`), place it under `/usr/lib/`.
   * Otherwise, inference will automatically fall back to CPU.
 
---
 
## Project Structure
 
```
.
├── test3.py                                          # Main script (e.g., live camera inference)
├── posenet_mobilenet_v1_075_353_481_quant.tflite     # Quantized TFLite model

```
 
* **`models/posenet_mobilenet_v1_075_353_481_quant.tflite`**
  Quantized MobileNet V1 PoseNet TensorFlow Lite model.
* **`test3.py`**
  Python script that:
 
  1. Loads the TFLite model (with optional NPU delegate).
  2. Captures frames from a webcam.
  3. Runs pose estimation (heatmaps + offsets).
  4. Decodes keypoint locations & confidence scores.
  5. Uses DBSCAN to select the main person.
  6. Draws keypoints & skeleton on the video feed.
 
---
 
## Usage
 
1. **Activate your virtual environment (if not already)**
 
   ```bash
   source venv/bin/activate
   ```
 
2. **Run the inference script**
 
   ```bash
   python3 test3.py
   ```
 
   * The script will attempt to load the NPU delegate library (`/usr/lib/libvx_delegate.so`).
   * If the delegate fails to load, it prints a warning and falls back to CPU.
   * A window titled **“PoseNet (Single Person)”** will open showing the webcam feed with drawn keypoints & skeleton.
   * Press **`q`** in the display window to exit and close the camera.
 
3. **Adjust parameters (optional)**
 
   * Inside `test3.py`, you can modify:
 
     * `min_score` in `select_main_person(...)` (default 0.4)
     * `threshold` in `draw_prediction_on_image(...)` (default 0.3)
     * DBSCAN parameters: `eps` and `min_samples` for clustering.
 
---
 
## How It Works
 
### 1. Loading the TFLite Model (with NPU Delegate)
 
```python
NPU_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"
try:
    delegate = tflite.load_delegate(NPU_DELEGATE_PATH)
    interpreter = tflite.Interpreter(
        model_path=model_path, experimental_delegates=[delegate]
    )
    print("Running on NPU")
except Exception as e:
    print(f"Failed to load NPU delegate: {e}")
    print("Falling back to CPU execution.")
    interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```
 
* Attempts to load an NPU delegate library.
* If successful, inference will run on the accelerator.
* Otherwise, interpreter runs on the CPU.
 
### 2. Preprocessing & Inference
 
```python
def preprocess_image(image, input_size):
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[np.newaxis, :].astype(np.uint8)
 
# In main loop:
input_image = preprocess_image(frame, input_size)
interpreter.set_tensor(input_details[0]["index"], input_image)
interpreter.invoke()
```
 
* Resizes each webcam frame to match the TFLite model’s expected input dimensions.
* Converts BGR→RGB and adds a batch dimension.
* Feeds the uint8 tensor into the interpreter and invokes inference.
 
### 3. Decoding Heatmaps & Offsets
 
```python
def dequantize(tensor, scale, zero_point):
    return scale * (tensor.astype(np.float32) - zero_point)
 
# After inference:
heatmaps_raw = interpreter.get_tensor(output_details[0]["index"])
heatmaps = dequantize(
    heatmaps_raw, output_details[0]["quantization"][0], output_details[0]["quantization"][1]
)
offsets_raw = interpreter.get_tensor(output_details[1]["index"])
offsets = dequantize(
    offsets_raw, output_details[1]["quantization"][0], output_details[1]["quantization"][1]
)
 
keypoints, scores = decode_pose(heatmaps, offsets)
```
 
* Retrieves quantized `heatmaps` and `offsets` from the model outputs.
* Dequantizes them to float32 using scale & zero point.
* Calls `decode_pose` to find the (x, y) location and confidence for each keypoint.
 
```python
def decode_pose(heatmaps, offsets, output_stride=16):
    heatmaps = heatmaps.squeeze()
    offsets = offsets.squeeze()
    num_keypoints = heatmaps.shape[-1]
 
    keypoints = np.zeros((num_keypoints, 2))
    scores = np.zeros(num_keypoints)
 
    for i in range(num_keypoints):
        hmap = heatmaps[:, :, i]
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        raw_score = hmap[y, x]
        max_score = np.max(hmap)
        scores[i] = raw_score / max_score if max_score > 0 else 0.0
 
        offset_y = offsets[y, x, i]
        offset_x = offsets[y, x, i + num_keypoints]
        keypoints[i, 0] = x * output_stride + offset_x
        keypoints[i, 1] = y * output_stride + offset_y
 
    return keypoints, scores
```
 
* For each of the 17 body keypoints:
 
  1. Finds the argmax in its heatmap → coarse coordinates.
  2. Normalizes the raw heatmap score to obtain a confidence between 0 and 1.
  3. Reads corresponding offsets to refine to full‐resolution coordinates.
 
### 4. Selecting the Main Person via DBSCAN
 
```python
def select_main_person(keypoints, scores, min_score=0.4):
    valid_mask = scores > min_score
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 3:
        return np.zeros_like(scores)
 
    valid_kps = keypoints[valid_indices]
    scaler = StandardScaler()
    scaled_kps = scaler.fit_transform(valid_kps)
 
    clustering = DBSCAN(eps=1, min_samples=3).fit(scaled_kps)
    labels = clustering.labels_
 
    cluster_scores = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_scores[label] = cluster_scores.get(label, 0) + scores[valid_indices[i]]
 
    if not cluster_scores:
        return np.zeros_like(scores)
 
    best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]
    filtered_scores = np.zeros_like(scores)
    for idx, label in zip(valid_indices, labels):
        if label == best_cluster:
            filtered_scores[idx] = scores[idx]
 
    return filtered_scores
```
 
* **Step 1:** Only keep keypoints whose confidence > `min_score` (default 0.4).
* **Step 2:** Standardize their (x, y) coordinates.
* **Step 3:** Run DBSCAN (eps = 1, min\_samples = 3) on standardized coordinates.
* **Step 4:** Sum raw confidence scores per cluster to identify which cluster likely belongs to one person.
* **Step 5:** Return a new score vector that is non‐zero only for keypoints of the highest‐scoring cluster.
 
### 5. Drawing Keypoints & Skeleton
 
```python
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255), (0, 2): (0, 255, 255),
    (1, 3): (255, 0, 255), (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255), (0, 6): (0, 255, 255),
    (5, 7): (255, 0, 255), (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255), (8, 10): (0, 255, 255),
    (5, 6): (255, 255, 0), (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255), (11, 12): (255, 255, 0),
    (11, 13): (255, 0, 255), (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255), (14, 16): (0, 255, 255)
}
 
def draw_prediction_on_image(image, keypoints, scores, threshold=0.3):
    # Draw circles for each keypoint above `threshold`
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > threshold:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
 
    # Draw skeleton edges when both endpoints exceed `threshold`
    for (i1, i2), color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if scores[i1] > threshold and scores[i2] > threshold:
            pt1 = tuple(keypoints[i1].astype(int))
            pt2 = tuple(keypoints[i2].astype(int))
            cv2.line(image, pt1, pt2, color, 2)
 
    return image
```
 
* Circles are drawn at each keypoint with confidence > `threshold` (default 0.3).
* Skeleton edges (as defined in `KEYPOINT_EDGE_INDS_TO_COLOR`) are drawn in magenta/cyan/yellow when both corresponding keypoints exceed the threshold.
 
---
 
## Dependencies
 
Install these packages via `pip`:
 
```bash
pip install numpy opencv-python-headless tflite-runtime scikit-learn
```
 
* **numpy** – Array operations and numerical computations.
* **opencv-python-headless** – Image capture and drawing functions (no GUI). If you want to use OpenCV’s GUI windows (imshow), instead install `opencv-python`:
 
  ```bash
  pip install opencv-python
  ```
* **tflite-runtime** – Lightweight TFLite interpreter (no full TensorFlow dependency).
* **scikit-learn** – Implements DBSCAN and `StandardScaler`.
 
> **NPU Delegate Library** (Optional)
> If you have a hardware accelerator (NPU/EdgeTPU) that requires a delegate, place the shared library (e.g., `libvx_delegate.so`) under `/usr/lib/`. The script will attempt to load it at runtime; if loading fails, it automatically runs on CPU.
 
---
 
## Troubleshooting
 
* **“Failed to load NPU delegate”**
 
  * Ensure that `libvx_delegate.so` (or your board’s delegate) exists in `/usr/lib/` and has correct permissions.
  * Verify the correct path in `NPU_DELEGATE_PATH` inside `test3.py`.
  * If you do not have an NPU, ignore this warning—the script will use CPU.
 
* **OpenCV cannot open camera**
 
  * Check that your webcam is connected and recognized by your OS (try `ls /dev/video*`).
  * If you are on a headless server without a physical camera, consider using a video file by replacing `cv2.VideoCapture(0)` with `cv2.VideoCapture("path/to/video.mp4")`.
 
* **Performance is slow on CPU**
 
  * Lower the input resolution (adjust `input_size`).
  * Reduce the frame processing rate (e.g., skip frames).
  * Obtain and configure an appropriate NPU delegate for hardware acceleration.
 
* **DBSCAN clusters everything as noise**
 
  * Tweak `eps` (currently `1.0`) and `min_samples` in the `DBSCAN(...)` call.
  * Adjust `min_score` in `select_main_person` to include more keypoints or fewer low‐confidence ones.
 
---
 
## License
 
This project is released under the [MIT License](LICENSE). Feel free to reuse and modify for your own PoseNet experiments.
 
```text
MIT License
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     
copies of the Software, and to permit persons to whom the Software is          
furnished to do so, subject to the following conditions:                       
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.                                 
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.                                                                       
```
 
---
 
> **Enjoy experimenting with PoseNet & NPU acceleration!**
