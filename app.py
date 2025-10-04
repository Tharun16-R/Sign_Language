import os
import io
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import numpy as np
from gtts import gTTS

from model.infer import load_model, predict_sequence

app = Flask(__name__)
CORS(app)  # allow all origins; tighten as needed

# Config
SEQ_LEN = int(os.environ.get("SEQ_LEN", 30))
NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 63))  # 21 landmarks * (x,y,z) single hand
LABELS = os.environ.get("LABELS", "HELLO,THANKS,ILOVEYOU").split(",")
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "model/weights/best_model.pth")

# Ensure weights dir exists
os.makedirs(os.path.dirname(MODEL_WEIGHTS), exist_ok=True)

model = load_model(num_features=NUM_FEATURES, num_classes=len(LABELS), weights_path=MODEL_WEIGHTS)


@app.route("/")
def index():
    return render_template("index.html", labels=LABELS, seq_len=SEQ_LEN)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    seq = data.get("sequence")
    if not seq:
        return jsonify({"error": "Missing 'sequence'"}), 400
    arr = np.array(seq, dtype=np.float32)
    # Expect shape (T, F)
    if arr.ndim != 2:
        return jsonify({"error": "sequence must be 2D (T,F)"}), 400
    # Pad/trim to SEQ_LEN
    T, F = arr.shape
    if F != NUM_FEATURES:
        return jsonify({"error": f"Expected feature size {NUM_FEATURES}, got {F}"}), 400
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, F), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    elif T > SEQ_LEN:
        arr = arr[-SEQ_LEN:]

    idx, probs = predict_sequence(model, arr)
    pred_label = LABELS[int(idx)]
    return jsonify({"prediction": pred_label, "index": int(idx), "probs": probs.tolist(), "labels": LABELS})


@app.post("/text_to_speech")
def text_to_speech():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return send_file(buf, mimetype="audio/mpeg", as_attachment=False, download_name="speech.mp3")


@app.get("/text_to_sign")
def text_to_sign():
    text = request.args.get("text", "")
    # Demo: map letters to sample video URLs under static/videos/
    mapping = []
    for ch in text.upper():
        if ch.isalpha():
            url = f"/static/videos/{ch}.mp4"  # provide your own videos
            mapping.append({"char": ch, "url": url})
        else:
            mapping.append({"char": ch, "url": None})
    return jsonify({"sequence": mapping})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    app.run(host=host, port=port, debug=True)
