"""
Student Emotion Detection System
Backend: Flask + OpenCV DNN (no TensorFlow / DeepFace required)

Emotion model: Mini-Xception trained on FER-2013 (Kaggle)
  - Weights: emotion_model.hdf5  (downloaded automatically on first run)
  - Face detector: OpenCV Haar Cascade (built-in)
"""

from flask import Flask, render_template, Response, jsonify, send_file
from flask_cors import CORS
import cv2
import json
import os
import datetime
import threading
import urllib.request
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
PROTO_PATH  = os.path.join(MODEL_DIR, "deploy.prototxt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.hdf5")

# ── Emotion labels (FER-2013 order) ──────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Neutral"]

# Classroom-friendly remapping
EMOTION_REMAP = {
    "Angry":     "Angry",
    "Disgusted": "Bored",
    "Scared":    "Confused",
    "Happy":     "Happy",
    "Sad":       "Sad",
    "Surprised": "Surprised",
    "Neutral":   "Neutral",
}

EMOTION_COLORS_BGR = {
    "Happy":     (100, 220,  0),
    "Sad":       ( 80,  80, 200),
    "Angry":     (  0,   0, 220),
    "Confused":  (220, 180,  0),
    "Bored":     (150,  80, 150),
    "Surprised": (  0, 200, 200),
    "Neutral":   (180, 180, 180),
}

# ── Global state ──────────────────────────────────────────────────────────────
camera       = None
camera_lock  = threading.Lock()
current_emotion = {
    "emotion": "Initializing...",
    "confidence": 0.0,
    "all_scores": {},
}
emotion_log  = []
LOG_FILE     = os.path.join(os.path.dirname(__file__), "emotion_log.json")
frame_counter = 0

# DNN models (loaded once)
face_net     = None
emotion_net  = None
haar_cascade = None

# ── Model download helpers ────────────────────────────────────────────────────

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def download_file(url, dest):
    if os.path.exists(dest):
        return
    print(f"[INFO] Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] Saved to {dest}")
    except Exception as e:
        print(f"[WARN] Download failed: {e}")


def load_models():
    """Load face detector and emotion classifier."""
    global face_net, emotion_net, haar_cascade

    ensure_model_dir()

    # ── Haar Cascade (always available in OpenCV) ──
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(haar_path)

    # ── Emotion model: Mini-Xception via OpenCV DNN ──
    # We use the ONNX version of the FER+ model (Microsoft) — no TF needed
    onnx_path = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
    onnx_url  = (
        "https://github.com/onnx/models/raw/main/validated/"
        "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    )
    download_file(onnx_url, onnx_path)

    if os.path.exists(onnx_path):
        try:
            emotion_net = cv2.dnn.readNetFromONNX(onnx_path)
            print("[INFO] Emotion model loaded (ONNX FER+).")
        except Exception as e:
            print(f"[WARN] Could not load ONNX model: {e}")
            emotion_net = None
    else:
        print("[WARN] Emotion model not found — running face-only mode.")
        emotion_net = None


# FER+ label order (matches the ONNX model output)
FERPLUS_LABELS = [
    "Neutral", "Happy", "Surprised", "Sad",
    "Angry", "Disgusted", "Scared", "Contempt",
]

FERPLUS_REMAP = {
    "Neutral":   "Neutral",
    "Happy":     "Happy",
    "Surprised": "Surprised",
    "Sad":       "Sad",
    "Angry":     "Angry",
    "Disgusted": "Bored",
    "Scared":    "Confused",
    "Contempt":  "Bored",
}

# ── Log helpers ───────────────────────────────────────────────────────────────

def load_log():
    global emotion_log
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                emotion_log = json.load(f)
        except Exception:
            emotion_log = []
    else:
        emotion_log = []


def save_log():
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(emotion_log, f, indent=2)
    except IOError as e:
        print(f"[WARN] Could not save log: {e}")


# ── Camera helpers ────────────────────────────────────────────────────────────

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


# ── Emotion analysis ──────────────────────────────────────────────────────────

def analyze_frame(frame):
    """Detect faces and classify emotions using OpenCV DNN."""
    global current_emotion

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) == 0:
        current_emotion = {"emotion": "No Face", "confidence": 0.0, "all_scores": {}}
        return

    # Use the largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi = gray[y:y+h, x:x+w]

    if emotion_net is None:
        # Fallback: no model loaded
        current_emotion = {"emotion": "Neutral", "confidence": 50.0, "all_scores": {}}
        return

    try:
        # FER+ ONNX expects: 1×1×64×64 float32, pixel values 0-255
        resized = cv2.resize(face_roi, (64, 64))
        blob    = cv2.dnn.blobFromImage(
            resized, scalefactor=1.0, size=(64, 64),
            mean=(0,), swapRB=False
        )
        emotion_net.setInput(blob)
        preds = emotion_net.forward()[0]          # shape (8,)

        # Softmax
        e_x   = np.exp(preds - np.max(preds))
        probs = e_x / e_x.sum()                  # 0-1

        top_idx   = int(np.argmax(probs))
        raw_label = FERPLUS_LABELS[top_idx]
        mapped    = FERPLUS_REMAP.get(raw_label, raw_label)
        confidence = float(probs[top_idx]) * 100

        all_scores = {
            FERPLUS_REMAP.get(FERPLUS_LABELS[i], FERPLUS_LABELS[i]): round(float(probs[i]) * 100, 2)
            for i in range(len(FERPLUS_LABELS))
        }

        current_emotion = {
            "emotion":    mapped,
            "confidence": round(confidence, 2),
            "all_scores": all_scores,
        }

        entry = {
            "timestamp":  datetime.datetime.now().isoformat(),
            "emotion":    mapped,
            "confidence": round(confidence, 2),
        }
        emotion_log.append(entry)
        if len(emotion_log) % 30 == 0:
            save_log()

    except Exception as e:
        print(f"[WARN] Emotion analysis error: {e}")
        current_emotion = {"emotion": "No Face", "confidence": 0.0, "all_scores": {}}


# ── Frame overlay ─────────────────────────────────────────────────────────────

def draw_overlay(frame):
    h, w   = frame.shape[:2]
    label  = current_emotion.get("emotion", "")
    conf   = current_emotion.get("confidence", 0.0)
    color  = EMOTION_COLORS_BGR.get(label, (255, 255, 255))

    # Draw face bounding boxes
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text = f"{label}  ({conf:.1f}%)"
    cv2.putText(frame, text, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (w - 245, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ── MJPEG stream ──────────────────────────────────────────────────────────────

def generate_frames():
    global frame_counter
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        frame_counter += 1
        if frame_counter % 8 == 0:          # analyze every 8th frame
            analyze_frame(frame.copy())
        frame = draw_overlay(frame)
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/emotion_status")
def emotion_status():
    return jsonify(current_emotion)


@app.route("/emotion_log")
def get_emotion_log():
    return jsonify(emotion_log[-200:])


@app.route("/emotion_summary")
def emotion_summary():
    counts = {}
    for entry in emotion_log:
        e = entry.get("emotion", "Unknown")
        counts[e] = counts.get(e, 0) + 1
    total   = sum(counts.values()) or 1
    summary = {k: {"count": v, "percent": round(v / total * 100, 1)}
               for k, v in counts.items()}
    return jsonify({"summary": summary, "total": total})


@app.route("/clear_log", methods=["POST"])
def clear_log():
    global emotion_log
    emotion_log = []
    save_log()
    return jsonify({"status": "cleared"})


@app.route("/export_log")
def export_log():
    save_log()
    if os.path.exists(LOG_FILE):
        return send_file(LOG_FILE, as_attachment=True,
                         download_name="emotion_log.json")
    return jsonify({"error": "No log file found"}), 404


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    release_camera()
    return jsonify({"status": "camera released"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_log()
    load_models()
    print("=" * 55)
    print("  Student Emotion Detection System")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 55)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
