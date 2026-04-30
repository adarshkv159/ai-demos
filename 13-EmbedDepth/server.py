"""
MiDaS v2.1 Small — Inference Server
Runs on phyBOARD-pollux (imx8m plus aarch64)
Supports NPU (VX delegate) with CPU fallback
"""

import cv2
import numpy as np
import base64
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_RUNTIME = False

# ── Model path ─────────────────────────────────────────────────
MODEL_PATH = "midas_v2.1_small_quant_recalib.tflite"

# ── MiDaS INT8 quantization params ────────────────────────────
INPUT_SCALE  = 0.01865844801068306
INPUT_ZERO   = -14
OUTPUT_SCALE = 10.557392120361328
OUTPUT_ZERO  = -128
IMG_SIZE     = 256
IMG_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Global state ───────────────────────────────────────────────
interpreter  = None
use_npu      = False
npu_available = False

# ── Interpreter loader ─────────────────────────────────────────
def load_interpreter(npu=True):
    global use_npu, npu_available

    if npu and TFLITE_RUNTIME:
        try:
            print("[SERVER] Trying NPU (VX delegate)…")
            delegate = tflite.load_delegate("/usr/lib/libvx_delegate.so")
            interp = tflite.Interpreter(
                model_path=MODEL_PATH,
                experimental_delegates=[delegate]
            )
            interp.allocate_tensors()
            use_npu = True
            npu_available = True
            print("[SERVER]  NPU delegate loaded")
            return interp
        except Exception as e:
            print(f"[SERVER]   NPU failed: {e}")
            npu_available = False

    print("[SERVER] Using CPU (4 threads)")
    if TFLITE_RUNTIME:
        interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    else:
        interp = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interp.allocate_tensors()
    use_npu = False
    return interp

# ── Preprocessing ──────────────────────────────────────────────
def preprocess(rgba_bytes):
    """rgba_bytes: raw RGBA bytes from canvas (256×256×4)"""
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE, 4)
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    rgb = (rgb - IMG_MEAN) / IMG_STD
    return rgb

def quantize_input(img):
    q = img / INPUT_SCALE + INPUT_ZERO
    q = np.clip(q, -128, 127).astype(np.int8)
    return q[np.newaxis, ...]          # [1, 256, 256, 3]

def dequantize_output(q):
    return OUTPUT_SCALE * (q.astype(np.float32) - OUTPUT_ZERO)

def run_inference(rgba_bytes):
    img    = preprocess(rgba_bytes)
    q_in   = quantize_input(img)

    inp    = interpreter.get_input_details()
    out    = interpreter.get_output_details()

    interpreter.set_tensor(inp[0]['index'], q_in)
    interpreter.invoke()

    q_out  = interpreter.get_tensor(out[0]['index'])[0]   # [256, 256]
    depth  = dequantize_output(q_out)

    # Normalize to [0, 1]
    mn, mx = depth.min(), depth.max()
    depth  = (depth - mn) / max(mx - mn, 1e-8)

    return depth.flatten().tolist()

# ── HTTP handler ───────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # suppress per-request logs

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self.send_json(200, {
                'ok': interpreter is not None,
                'npu': use_npu,
                'npu_available': npu_available,
                'model': MODEL_PATH,
                'size': IMG_SIZE
            })
        elif self.path == '/':
            # Serve the HTML file
            html_path = os.path.join(os.path.dirname(__file__), 'midas_pointcloud.html')
            try:
                with open(html_path, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(body))
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_json(404, {'error': 'midas_pointcloud.html not found'})
        else:
            self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        global interpreter
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)

        # ── /infer — main inference endpoint ──────────────────
        if self.path == '/infer':
            if interpreter is None:
                self.send_json(503, {'error': 'model not loaded'})
                return
            try:
                data       = json.loads(body)
                rgba_bytes = base64.b64decode(data['rgba'])
                depth      = run_inference(rgba_bytes)
                self.send_json(200, {'depth': depth})
            except Exception as e:
                self.send_json(500, {'error': str(e)})

        # ── /set_backend — switch NPU ↔ CPU at runtime ────────
        elif self.path == '/set_backend':
            try:
                data   = json.loads(body)
                want   = data.get('npu', False)
                interpreter = load_interpreter(npu=want)
                self.send_json(200, {
                    'ok': True,
                    'npu': use_npu,
                    'npu_available': npu_available
                })
            except Exception as e:
                self.send_json(500, {'error': str(e)})

        else:
            self.send_json(404, {'error': 'not found'})


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"[SERVER]   Model not found: {MODEL_PATH}")
        exit(1)

    print(f"[SERVER] Loading model: {MODEL_PATH}")
    interpreter = load_interpreter(npu=True)   # try NPU first

    HOST, PORT = '0.0.0.0', 5001
    print(f"[SERVER]   Listening on http://{HOST}:{PORT}")
    print(f"[SERVER]     Backend: {'NPU ' if use_npu else 'CPU'}")
    print(f"[SERVER]     Open browser → http://localhost:{PORT}")

    HTTPServer((HOST, PORT), Handler).serve_forever()
