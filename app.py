"""
Student Emotion Detection System
Backend: Flask + OpenCV DNN (browser-based webcam, deployable to Render/cloud)
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import cv2
import json
import os
import datetime
import numpy as np
import base64
import urllib.request

app = Flask(__name__)
CORS(app)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
ONNX_PATH  = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")
ONNX_URL   = (
    "https://github.com/onnx/models/raw/refs/heads/main/validated/"
    "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# ── Emotion metadata ──────────────────────────────────────────────────────────
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

# ── Global state ──────────────────────────────────────────────────────────────
emotion_log  = []
LOG_FILE     = os.path.join(os.path.dirname(__file__), "emotion_log.json")
emotion_net  = None
haar_cascade = None

# ── Model loader ──────────────────────────────────────────────────────────────

def load_models():
    global emotion_net, haar_cascade

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Haar Cascade
    haar_path    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(haar_path)

    # Download ONNX model if missing
    if not os.path.exists(ONNX_PATH):
        print("[INFO] Downloading emotion model...")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(ONNX_URL, ONNX_PATH)
            size = os.path.getsize(ONNX_PATH)
            print(f"[INFO] Model downloaded ({size} bytes).")
            if size < 100000:
                os.remove(ONNX_PATH)
                print("[WARN] Downloaded file too small, removing.")
        except Exception as e:
            print(f"[WARN] Download failed: {e}")

    if os.path.exists(ONNX_PATH):
        try:
            emotion_net = cv2.dnn.readNetFromONNX(ONNX_PATH)
            print("[INFO] Emotion model loaded.")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")


# ── Log helpers ───────────────────────────────────────────────────────────────

def load_log():
    global emotion_log
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                emotion_log = json.load(f)
        except Exception:
            emotion_log = []


def save_log():
    try:
        with open(LOG_FILE, "w") as f:
            json.dump(emotion_log, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save log: {e}")


# ── Emotion analysis ──────────────────────────────────────────────────────────

def analyze_image(img_array):
    """Analyze a BGR numpy image and return emotion result dict."""
    if haar_cascade is None:
        return {"emotion": "Model Error", "confidence": 0.0, "all_scores": {}}

    gray  = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )

    if len(faces) == 0:
        return {"emotion": "No Face", "confidence": 0.0, "all_scores": {}}

    # Largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi   = gray[y:y+h, x:x+w]

    if emotion_net is None:
        return {"emotion": "Neutral", "confidence": 50.0, "all_scores": {}}

    try:
        resized = cv2.resize(face_roi, (64, 64))
        blob    = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0,), swapRB=False)
        emotion_net.setInput(blob)
        preds   = emotion_net.forward()[0]

        e_x   = np.exp(preds - np.max(preds))
        probs = e_x / e_x.sum()

        top_idx    = int(np.argmax(probs))
        raw_label  = FERPLUS_LABELS[top_idx]
        mapped     = FERPLUS_REMAP.get(raw_label, raw_label)
        confidence = float(probs[top_idx]) * 100

        all_scores = {
            FERPLUS_REMAP.get(FERPLUS_LABELS[i], FERPLUS_LABELS[i]):
                round(float(probs[i]) * 100, 2)
            for i in range(len(FERPLUS_LABELS))
        }

        # Log entry
        entry = {
            "timestamp":  datetime.datetime.now().isoformat(),
            "emotion":    mapped,
            "confidence": round(confidence, 2),
        }
        emotion_log.append(entry)
        if len(emotion_log) % 30 == 0:
            save_log()

        return {
            "emotion":    mapped,
            "confidence": round(confidence, 2),
            "all_scores": all_scores,
        }

    except Exception as e:
        print(f"[WARN] Analysis error: {e}")
        return {"emotion": "No Face", "confidence": 0.0, "all_scores": {}}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Receive a base64 image frame from the browser and return emotion JSON."""
    try:
        data     = request.get_json()
        img_data = data.get("image", "")

        # Strip data URL prefix if present
        if "," in img_data:
            img_data = img_data.split(",")[1]

        img_bytes  = base64.b64decode(img_data)
        img_array  = np.frombuffer(img_bytes, dtype=np.uint8)
        frame      = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"emotion": "No Face", "confidence": 0.0, "all_scores": {}})

        result = analyze_image(frame)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] /analyze: {e}")
        return jsonify({"emotion": "Error", "confidence": 0.0, "all_scores": {}}), 500


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


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Auto-load on import (needed for gunicorn) ─────────────────────────────────
load_log()
load_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  Student Emotion Detection System")
    print(f"  Open http://127.0.0.1:{port} in your browser")
    print("=" * 55)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)

