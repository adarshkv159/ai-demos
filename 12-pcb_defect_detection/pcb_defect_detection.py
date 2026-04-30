#!/usr/bin/env python3
"""
PCB Defect Detection — GTK3 GUI
White theme, live inference, crop gallery for detected defects (conf >= 0.60)
"""

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

import cv2
import numpy as np
import time
import threading
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH    = "best_saved_model/pcb_defect_detection_yolov8n_full_integer_quant.tflite"
DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

PCB_CLASSES = [
    'missing_hole', 'mouse_bite', 'open_circuit',
    'short', 'spur', 'spurious_copper'
]

# BGR colors for OpenCV drawing
COLORS_BGR = [
    (0,   0,   255),
    (0,   200, 0  ),
    (255, 80,  0  ),
    (0,   200, 200),
    (200, 0,   200),
    (0,   160, 255),
]

# Hex colors for GTK labels
COLORS_HEX = [
    "#FF2222", "#00C800", "#FF5000",
    "#00C8C8", "#C800C8", "#00A0FF",
]

CONF_THR_DISPLAY = 0.60   # threshold for crop gallery
IOU_THR          = 0.45
CROP_SIZE        = 96     # px for defect thumbnails

# ─── Camera modes ─────────────────────────────────────────────────────────────

CAMERAS = {
    "Webcam (V4L2)": {
        "type": "v4l2",
        "id":   0,
        "conf": 0.35,
    },
    "VM-016 (CSI / GStreamer)": {
        "type": "gstreamer",
        "pipeline": (
            "v4l2src device=/dev/video-isp-csi1 ! "
            "video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink max-buffers=1 drop=true sync=false"
        ),
        "conf": 0.35,
    },
}

# ─── Model helpers (same as original scripts) ─────────────────────────────────

def load_model():
    from tflite_runtime.interpreter import Interpreter, load_delegate
    delegates = []
    try:
        delegates.append(load_delegate(DELEGATE_PATH))
        print("NPU delegate loaded — hardware acceleration ENABLED.")
    except Exception as e:
        print(f"NPU delegate failed ({e}), using CPU.")

    interp = Interpreter(model_path=MODEL_PATH, experimental_delegates=delegates)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale,  in_zp  = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    input_size = inp["shape"][1]
    in_dtype   = inp["dtype"]
    return interp, inp, out, input_size, (in_scale, in_zp), (out_scale, out_zp), in_dtype


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pl = (size - nw) // 2
    pt = (size - nh) // 2
    img = cv2.copyMakeBorder(img, pt, size - nh - pt, pl, size - nw - pl,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (pl, pt, scale)


def preprocess(frame, input_size, in_dtype, in_scale, in_zp):
    img, pad_info = letterbox(frame, input_size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if in_dtype == np.int8:
        tensor = (rgb.astype(np.float32) / 255.0 / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    elif in_dtype == np.uint8:
        tensor = (rgb.astype(np.float32) / 255.0 / in_scale + in_zp).clip(0, 255).astype(np.uint8)
    else:
        tensor = rgb.astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0), pad_info


def dequantize(tensor, scale, zp):
    return (tensor.astype(np.float32) - zp) * scale


def postprocess(raw_out, pad_info, orig_shape, out_scale, out_zp, conf_thr, iou_thr, input_size=640):
    out = dequantize(raw_out[0], out_scale, out_zp)
    boxes_xywh   = out[:4, :].T * input_size
    class_scores = out[4:, :].T
    scores    = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)
    mask = scores >= conf_thr
    boxes_xywh = boxes_xywh[mask]
    scores     = scores[mask]
    class_ids  = class_ids[mask]
    if len(scores) == 0:
        return [], [], []
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    pl, pt, scale = pad_info
    oh, ow = orig_shape[:2]
    x1 = ((x1 - pl) / scale).clip(0, ow)
    y1 = ((y1 - pt) / scale).clip(0, oh)
    x2 = ((x2 - pl) / scale).clip(0, ow)
    y2 = ((y2 - pt) / scale).clip(0, oh)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = []
    for cid in np.unique(class_ids):
        m   = class_ids == cid
        idx = np.where(m)[0]
        b   = boxes_xyxy[m]
        s   = scores[m]
        xywh = [[bx[0], bx[1], bx[2]-bx[0], bx[3]-bx[1]] for bx in b]
        nms  = cv2.dnn.NMSBoxes(xywh, s.tolist(), conf_thr, iou_thr)
        if len(nms) > 0:
            keep.extend(idx[nms.flatten()])
    return boxes_xyxy[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()


def draw_boxes(frame, boxes, scores, class_ids, fps):
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS_BGR[cid]
        label = f"{PCB_CLASSES[cid]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    return frame


def frame_to_pixbuf(frame_bgr, width, height):
    """BGR numpy array -> GdkPixbuf scaled to (width, height)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    h, w, ch = rgb.shape
    return GdkPixbuf.Pixbuf.new_from_data(
        rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8,
        w, h, w * ch, None, None
    )


def crop_to_pixbuf(frame_bgr, box, size=CROP_SIZE):
    x1, y1, x2, y2 = map(int, box)
    # add 8px margin
    h, w = frame_bgr.shape[:2]
    margin = 8
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.full((size, size, 3), 200, dtype=np.uint8)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return GdkPixbuf.Pixbuf.new_from_data(
        rgb.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8,
        size, size, size * 3, None, None
    )


# ─── GTK Application ──────────────────────────────────────────────────────────

CSS = """
* { font-family: "DejaVu Sans", sans-serif; }

window {
    background-color: #FFFFFF;
}

/* ── Top bar ── */
#topbar {
    background-color: #FAFAFA;
    border-bottom: 1px solid #E0E0E0;
    padding: 10px 16px;
}
#app-title {
    font-size: 15px;
    font-weight: bold;
    color: #1A1A1A;
    letter-spacing: 0.5px;
}
#status-label {
    font-size: 12px;
    color: #888888;
}

/* ── Camera selector ── */
#cam-label {
    font-size: 12px;
    color: #555555;
    margin-right: 6px;
}
combobox button {
    background-color: #F5F5F5;
    border: 1px solid #D0D0D0;
    border-radius: 5px;
    padding: 4px 10px;
    color: #222222;
    font-size: 12px;
}
combobox button:hover {
    background-color: #EBEBEB;
}

/* ── Start / Stop buttons ── */
#btn-start {
    background-color: #2E7D32;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 6px 20px;
    font-size: 13px;
    font-weight: bold;
}
#btn-start:hover   { background-color: #1B5E20; }
#btn-start:disabled { background-color: #A5D6A7; }

#btn-stop {
    background-color: #FFFFFF;
    color: #D32F2F;
    border: 1px solid #D32F2F;
    border-radius: 6px;
    padding: 6px 20px;
    font-size: 13px;
    font-weight: bold;
}
#btn-stop:hover    { background-color: #FEECEC; }
#btn-stop:disabled { color: #CCCCCC; border-color: #CCCCCC; }

/* ── Save button ── */
#btn-save {
    background-color: #FFFFFF;
    color: #1A73E8;
    border: 1px solid #1A73E8;
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 12px;
}
#btn-save:hover { background-color: #E8F0FE; }

/* ── Video area ── */
#video-frame {
    background-color: #F0F0F0;
    border: 1px solid #E0E0E0;
    border-radius: 8px;
}
#video-placeholder {
    color: #AAAAAA;
    font-size: 13px;
}

/* ── Stats bar ── */
#stats-bar {
    background-color: #FAFAFA;
    border-top: 1px solid #EEEEEE;
    padding: 5px 16px;
}
#fps-label   { font-size: 12px; color: #1A73E8; font-weight: bold; }
#count-label { font-size: 12px; color: #D32F2F; font-weight: bold; }
#thresh-label { font-size: 11px; color: #888888; }

/* ── Defect panel ── */
#defect-panel-title {
    font-size: 13px;
    font-weight: bold;
    color: #1A1A1A;
    padding: 10px 14px 4px;
}
#defect-panel-sub {
    font-size: 11px;
    color: #AAAAAA;
    padding: 0 14px 8px;
}
#defect-scroll {
    background-color: #FFFFFF;
    border-top: 1px solid #EEEEEE;
}

/* ── Individual defect card ── */
.defect-card {
    background-color: #FAFAFA;
    border: 1px solid #E8E8E8;
    border-radius: 8px;
    margin: 4px;
    padding: 6px;
}
.defect-label {
    font-size: 11px;
    font-weight: bold;
    color: #1A1A1A;
}
.defect-score {
    font-size: 10px;
    color: #666666;
}
.defect-time {
    font-size: 9px;
    color: #AAAAAA;
}
"""


class PCBApp(Gtk.Window):

    def __init__(self):
        super().__init__(title="PCB Defect Detection")
        self.set_default_size(1100, 720)
        self.set_resizable(True)

        # State
        self._running    = False
        self._cap        = None
        self._interp     = None
        self._thread     = None
        self._fps_avg    = 0.0
        self._frame_lock = threading.Lock()
        self._latest_frame   = None
        self._latest_detects = []   # list of (box, score, cid, crop_pb)

        # Apply CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(CSS.encode("utf-8"))
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._build_ui()
        self.connect("destroy", self._on_destroy)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(root)

        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        topbar.set_name("topbar")
        topbar.set_hexpand(True)

        title = Gtk.Label(label="PCB Defect Detection")
        title.set_name("app-title")
        topbar.pack_start(title, False, False, 0)

        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        topbar.pack_start(sep, False, False, 4)

        cam_label = Gtk.Label(label="Camera:")
        cam_label.set_name("cam-label")
        topbar.pack_start(cam_label, False, False, 0)

        self._cam_combo = Gtk.ComboBoxText()
        for name in CAMERAS:
            self._cam_combo.append_text(name)
        self._cam_combo.set_active(0)
        topbar.pack_start(self._cam_combo, False, False, 0)

        # spacer
        topbar.pack_start(Gtk.Box(), True, True, 0)

        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_name("status-label")
        topbar.pack_start(self._status_label, False, False, 0)

        sep2 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        topbar.pack_start(sep2, False, False, 4)

        self._btn_start = Gtk.Button(label="Start")
        self._btn_start.set_name("btn-start")
        self._btn_start.connect("clicked", self._on_start)
        topbar.pack_start(self._btn_start, False, False, 0)

        self._btn_stop = Gtk.Button(label="Stop")
        self._btn_stop.set_name("btn-stop")
        self._btn_stop.set_sensitive(False)
        self._btn_stop.connect("clicked", self._on_stop)
        topbar.pack_start(self._btn_stop, False, False, 0)

        self._btn_save = Gtk.Button(label="Save")
        self._btn_save.set_name("btn-save")
        self._btn_save.connect("clicked", self._on_save)
        topbar.pack_start(self._btn_save, False, False, 0)

        root.pack_start(topbar, False, False, 0)

        # ── Main content ─────────────────────────────────────────────────────
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        content.set_vexpand(True)
        root.pack_start(content, True, True, 0)

        # Left: video
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        left.set_hexpand(True)
        content.pack_start(left, True, True, 0)

        video_frame = Gtk.AspectFrame(ratio=16/9, obey_child=False)
        video_frame.set_name("video-frame")
        video_frame.set_hexpand(True)
        video_frame.set_vexpand(True)
        video_frame.set_border_width(12)

        self._video_image = Gtk.Image()
        video_frame.add(self._video_image)

        # Placeholder overlay when not running
        self._placeholder = Gtk.Label(label="Select a camera and press Start")
        self._placeholder.set_name("video-placeholder")

        self._video_stack = Gtk.Stack()
        self._video_stack.add_named(self._placeholder, "placeholder")
        self._video_stack.add_named(video_frame, "video")
        self._video_stack.set_visible_child_name("placeholder")
        left.pack_start(self._video_stack, True, True, 0)

        # Stats bar
        stats = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=18)
        stats.set_name("stats-bar")

        self._fps_label = Gtk.Label(label="FPS: —")
        self._fps_label.set_name("fps-label")
        stats.pack_start(self._fps_label, False, False, 0)

        self._count_label = Gtk.Label(label="Defects: 0")
        self._count_label.set_name("count-label")
        stats.pack_start(self._count_label, False, False, 0)

        thresh_label = Gtk.Label(label=f"(gallery threshold ≥ {CONF_THR_DISPLAY:.0%})")
        thresh_label.set_name("thresh-label")
        stats.pack_start(thresh_label, False, False, 0)

        left.pack_start(stats, False, False, 0)

        # Right: defect panel
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        right.set_size_request(230, -1)
        right.get_style_context().add_class("right-panel")

        sep_v = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        content.pack_start(sep_v, False, False, 0)
        content.pack_start(right, False, False, 0)

        panel_title = Gtk.Label(label="Live Detections")
        panel_title.set_name("defect-panel-title")
        panel_title.set_halign(Gtk.Align.START)
        right.pack_start(panel_title, False, False, 0)

        panel_sub = Gtk.Label(label="Crops of current defects")
        panel_sub.set_name("defect-panel-sub")
        panel_sub.set_halign(Gtk.Align.START)
        right.pack_start(panel_sub, False, False, 0)

        scroll = Gtk.ScrolledWindow()
        scroll.set_name("defect-scroll")
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)

        self._defect_box = Gtk.FlowBox()
        self._defect_box.set_max_children_per_line(2)
        self._defect_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._defect_box.set_homogeneous(True)
        self._defect_box.set_column_spacing(4)
        self._defect_box.set_row_spacing(4)
        self._defect_box.set_margin_start(8)
        self._defect_box.set_margin_end(8)
        self._defect_box.set_margin_top(4)
        self._defect_box.set_margin_bottom(8)

        scroll.add(self._defect_box)
        right.pack_start(scroll, True, True, 0)

        self.show_all()

    # ── Event handlers ───────────────────────────────────────────────────────

    def _on_start(self, _btn):
        cam_name = self._cam_combo.get_active_text()
        cam_cfg  = CAMERAS[cam_name]
        self._conf_thr = cam_cfg["conf"]

        # Open camera
        if cam_cfg["type"] == "gstreamer":
            self._cap = cv2.VideoCapture(cam_cfg["pipeline"], cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(cam_cfg["id"], cv2.CAP_V4L2)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self._cap.isOpened():
            self._set_status(f"❌  Cannot open: {cam_name}")
            return

        # Load model if first time
        if self._interp is None:
            self._set_status("Loading model…")
            try:
                (self._interp, self._inp_det, self._out_det,
                 self._input_size, self._in_q, self._out_q,
                 self._in_dtype) = load_model()
            except Exception as e:
                self._set_status(f"❌  Model error: {e}")
                self._cap.release()
                return

        self._running = True
        self._fps_avg = 0.0
        self._btn_start.set_sensitive(False)
        self._btn_stop.set_sensitive(True)
        self._cam_combo.set_sensitive(False)
        self._video_stack.set_visible_child_name("video")
        self._set_status(f"Running — {cam_name}")

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        # Refresh UI ~30 fps
        self._ui_timer = GLib.timeout_add(33, self._refresh_ui)

    def _on_stop(self, _btn):
        self._running = False
        self._btn_start.set_sensitive(True)
        self._btn_stop.set_sensitive(False)
        self._cam_combo.set_sensitive(True)
        self._video_stack.set_visible_child_name("placeholder")
        self._set_status("Stopped")
        GLib.source_remove(self._ui_timer)

    def _on_save(self, _btn):
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        fname = f"pcb_capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(fname, frame)
        self._set_status(f"Saved {fname}")

    def _on_destroy(self, _win):
        self._running = False
        if self._cap:
            self._cap.release()
        Gtk.main_quit()

    # ── Inference thread ─────────────────────────────────────────────────────

    def _inference_loop(self):
        alpha    = 0.1
        in_sc,  in_zp  = self._in_q
        out_sc, out_zp = self._out_q

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            t0 = time.perf_counter()

            tensor, pad_info = preprocess(frame, self._input_size,
                                          self._in_dtype, in_sc, in_zp)
            self._interp.set_tensor(self._inp_det["index"], tensor)
            self._interp.invoke()
            raw_out = self._interp.get_tensor(self._out_det["index"])

            boxes, scores, class_ids = postprocess(
                raw_out, pad_info, frame.shape,
                out_sc, out_zp, self._conf_thr, IOU_THR, self._input_size
            )

            t1 = time.perf_counter()
            fps_inst  = 1.0 / max(t1 - t0, 1e-6)
            self._fps_avg = (fps_inst if self._fps_avg == 0
                             else alpha * fps_inst + (1 - alpha) * self._fps_avg)

            # Build annotated frame
            vis = draw_boxes(frame.copy(), boxes, scores, class_ids, self._fps_avg)

            # Build crops for gallery (only conf >= 0.60, one per class)
            seen_cids = {}
            for box, score, cid in zip(boxes, scores, class_ids):
                if score >= CONF_THR_DISPLAY:
                    if cid not in seen_cids or score > seen_cids[cid][1]:
                        seen_cids[cid] = (box, score)
            crops = []
            for cid, (box, score) in seen_cids.items():
                pb = crop_to_pixbuf(frame, box, CROP_SIZE)
                crops.append((box, score, cid, pb))

            with self._frame_lock:
                self._latest_frame   = vis
                self._latest_detects = crops

        if self._cap:
            self._cap.release()
            self._cap = None

    # ── UI refresh (main thread) ──────────────────────────────────────────────

    def _refresh_ui(self):
        if not self._running:
            return False

        with self._frame_lock:
            frame   = self._latest_frame
            detects = list(self._latest_detects)

        if frame is not None:
            alloc = self._video_image.get_allocation()
            w = max(alloc.width,  320)
            h = max(alloc.height, 180)
            pb = frame_to_pixbuf(frame, w, h)
            self._video_image.set_from_pixbuf(pb)

        # Stats
        self._fps_label.set_text(f"FPS: {self._fps_avg:.1f}")
        self._count_label.set_text(f"Defects: {len(detects)}")

        # Defect gallery
        for child in self._defect_box.get_children():
            self._defect_box.remove(child)

        for _box, score, cid, pb in detects:
            card = self._make_defect_card(pb, cid, score)
            self._defect_box.add(card)

        self._defect_box.show_all()
        return True   # keep timer alive

    def _make_defect_card(self, pixbuf, cid, score):
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        card.get_style_context().add_class("defect-card")

        img = Gtk.Image.new_from_pixbuf(pixbuf)
        card.pack_start(img, False, False, 0)

        name_lbl = Gtk.Label(label=PCB_CLASSES[cid].replace("_", " ").title())
        name_lbl.get_style_context().add_class("defect-label")
        name_lbl.set_halign(Gtk.Align.CENTER)
        # colour the label text via markup
        name_lbl.set_markup(
            f'<span foreground="{COLORS_HEX[cid]}" weight="bold" size="small">'
            f'{PCB_CLASSES[cid].replace("_", " ").title()}</span>'
        )
        card.pack_start(name_lbl, False, False, 0)

        score_lbl = Gtk.Label(label=f"{score:.2%}")
        score_lbl.get_style_context().add_class("defect-score")
        score_lbl.set_halign(Gtk.Align.CENTER)
        card.pack_start(score_lbl, False, False, 0)

        ts_lbl = Gtk.Label(label=datetime.now().strftime("%H:%M:%S"))
        ts_lbl.get_style_context().add_class("defect-time")
        ts_lbl.set_halign(Gtk.Align.CENTER)
        card.pack_start(ts_lbl, False, False, 0)

        return card

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, text):
        GLib.idle_add(self._status_label.set_text, text)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PCBApp()
    app.show_all()
    Gtk.main()
