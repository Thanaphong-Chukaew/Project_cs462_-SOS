import os, json, io, base64, threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_LOCK    = threading.Lock()
CURRENT_MODEL = {"model": None, "classes": [], "img_size": 64, "name": "none", "type": None}


# ── Model loading ─────────────────────────────────────────────────────────────

def _detect_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".joblib"):
        return "sklearn"
    if ext in (".h5", ".keras"):
        return "keras"
    return "unknown"


_PATCHED = False

def _patch_keras_compat():
    """Monkey-patch Keras so old .h5 models with extra config keys can load."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    try:
        from keras.src.ops.operation import Operation
        _orig = Operation.from_config.__func__

        @classmethod
        def _safe_from_config(cls, config):
            # Strip keys that older/newer Keras versions may not recognise
            drop_keys = {
                'renorm', 'renorm_clipping', 'renorm_momentum',
                'synchronized', 'quantization_config',
            }
            cleaned = {k: v for k, v in config.items() if k not in drop_keys}
            try:
                return _orig(cls, cleaned)
            except (TypeError, ValueError):
                # If still failing, try brute-force: keep only keys the __init__ accepts
                import inspect
                sig = inspect.signature(cls.__init__)
                valid = set(sig.parameters.keys()) - {'self'}
                if 'kwargs' in {p.name for p in sig.parameters.values()
                                if p.kind == inspect.Parameter.VAR_KEYWORD}:
                    return _orig(cls, cleaned)
                filtered = {k: v for k, v in cleaned.items() if k in valid}
                return _orig(cls, filtered)

        Operation.from_config = _safe_from_config
    except Exception:
        pass


def load_model_from_path(model_path, labels_path):
    mtype = _detect_type(model_path)

    with MODEL_LOCK:
        if mtype == "sklearn":
            import joblib
            m = joblib.load(model_path)
        elif mtype == "keras":
            import tensorflow as tf
            _patch_keras_compat()
            m = tf.keras.models.load_model(model_path)
        else:
            # ลองทั้งสองแบบ
            try:
                import joblib
                m = joblib.load(model_path)
                mtype = "sklearn"
            except Exception:
                import tensorflow as tf
                _patch_keras_compat()
                m = tf.keras.models.load_model(model_path)
                mtype = "keras"

        with open(labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        CURRENT_MODEL["model"]    = m
        CURRENT_MODEL["type"]     = mtype
        CURRENT_MODEL["classes"]  = meta["classes"]
        CURRENT_MODEL["img_size"] = meta.get("img_size", 64)
        CURRENT_MODEL["name"]     = os.path.basename(model_path)

    print(f"[OK] Loaded ({mtype}): {model_path}  classes={len(CURRENT_MODEL['classes'])} items")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _center_digit(arr, padding_ratio=0.12):
    """เหมือน center_digit ใน train_local.py — crop bounding box แล้ว center ใน square"""
    mask = arr > 20
    if not mask.any():
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    crop = arr[r0:r1+1, c0:c1+1]
    h, w = crop.shape
    side = max(h, w)
    pad = int(side * padding_ratio)
    side += pad * 2
    canvas = np.zeros((side, side), dtype=np.float32)
    y_off = pad + (side - pad*2 - h) // 2
    x_off = pad + (side - pad*2 - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop
    return canvas


def preprocess(b64_data_url, img_size, model_type):
    """
    ทำให้ตรงกับ training data:
      1. split canvas เป็นซ้าย/ขวาที่เส้นกลาง
      2. แต่ละฝั่ง: center digit ใน square (เหมือน center_digit ใน train)
      3. resize ซ้าย→ half×full, ขวา→ half×full
      4. concatenate → img_size × img_size
    """
    _, data = b64_data_url.split(",", 1)
    raw = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
    # Composite onto white background using alpha mask — mirrors training load_folder()
    # so transparent canvas pixels (L=0 after plain convert) don't invert to white
    bg = Image.new("L", raw.size, 255)
    bg.paste(raw.convert("L"), mask=raw.getchannel("A"))
    arr = 255.0 - np.array(bg, dtype=np.float32)   # invert: พื้นดำ ตัวขาว

    H, W = arr.shape
    mid = W // 2
    half = img_size // 2

    left  = _center_digit(arr[:, :mid])
    right = _center_digit(arr[:, mid:])

    pil_l = Image.fromarray(np.clip(left,  0, 255).astype(np.uint8)).resize((half, img_size), Image.LANCZOS)
    pil_r = Image.fromarray(np.clip(right, 0, 255).astype(np.uint8)).resize((half, img_size), Image.LANCZOS)

    combined = np.concatenate([np.array(pil_l, dtype=np.float32),
                               np.array(pil_r, dtype=np.float32)], axis=1) / 255.0

    if model_type == "sklearn":
        return combined.reshape(1, -1)
    return combined.reshape(1, img_size, img_size, 1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "webapp.html")


@app.route("/predict", methods=["POST"])
def predict():
    with MODEL_LOCK:
        if CURRENT_MODEL["model"] is None:
            return jsonify({"error": "Model not loaded. รัน python train_local.py ก่อน แล้ววาง model_v1.pkl ใน folder นี้"}), 503
        model    = CURRENT_MODEL["model"]
        classes  = CURRENT_MODEL["classes"]
        img_size = CURRENT_MODEL["img_size"]
        mtype    = CURRENT_MODEL["type"]

    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400

    try:
        tensor = preprocess(data["image"], img_size, mtype)

        if mtype == "sklearn":
            probs = model.predict_proba(tensor)[0]
        else:
            probs = model.predict(tensor, verbose=0)[0]

        idx = int(np.argmax(probs))
        return jsonify({
            "prediction":       classes[idx],
            "confidence":       float(probs[idx]),
            "all_probabilities": {c: float(p) for c, p in zip(classes, probs)},
            "model_used":       CURRENT_MODEL["name"],
            "model_type":       mtype,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload-model", methods=["POST"])
def upload_model():
    """Hot-swap model โดยไม่ต้อง restart server (รองรับทั้ง .pkl และ .h5)"""
    if "model" not in request.files:
        return jsonify({"error": "No model file attached"}), 400

    mf = request.files["model"]
    lf = request.files.get("labels")

    os.makedirs("uploads", exist_ok=True)
    model_path  = os.path.join("uploads", mf.filename)
    mf.save(model_path)

    labels_path = "class_labels.json"
    if lf:
        labels_path = os.path.join("uploads", "class_labels.json")
        lf.save(labels_path)

    if not os.path.exists(labels_path):
        return jsonify({"error": "class_labels.json not found. Upload alongside the model."}), 400

    try:
        load_model_from_path(model_path, labels_path)
        return jsonify({
            "success":    True,
            "model_name": mf.filename,
            "model_type": CURRENT_MODEL["type"],
            "classes":    CURRENT_MODEL["classes"],
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500


@app.route("/debug-preprocess", methods=["POST"])
def debug_preprocess():
    """ส่งรูปที่ผ่าน preprocess กลับมาเป็น base64 — ใช้ตรวจว่า model เห็นอะไร"""
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "Missing 'image'"}), 400
    img_size = CURRENT_MODEL.get("img_size", 64) or 64

    _, raw = data["image"].split(",", 1)
    img_rgba = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
    bg = Image.new("L", img_rgba.size, 255)
    bg.paste(img_rgba.convert("L"), mask=img_rgba.getchannel("A"))
    arr = 255.0 - np.array(bg, dtype=np.float32)

    H, W = arr.shape
    mid  = W // 2
    half = img_size // 2

    left  = _center_digit(arr[:, :mid])
    right = _center_digit(arr[:, mid:])

    pil_l = Image.fromarray(np.clip(left,  0,255).astype(np.uint8)).resize((half, img_size), Image.LANCZOS)
    pil_r = Image.fromarray(np.clip(right, 0,255).astype(np.uint8)).resize((half, img_size), Image.LANCZOS)
    combined = np.concatenate([np.array(pil_l), np.array(pil_r)], axis=1)

    # re-invert for display (digit=black on white)
    display = Image.fromarray((255 - combined).astype(np.uint8)).resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    display.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return jsonify({"debug_image": f"data:image/png;base64,{b64}"})


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_name": CURRENT_MODEL["name"],
        "model_type": CURRENT_MODEL["type"],
        "classes":    CURRENT_MODEL["classes"],
        "img_size":   CURRENT_MODEL["img_size"],
        "loaded":     CURRENT_MODEL["model"] is not None,
    })


# ── Startup ───────────────────────────────────────────────────────────────────

def try_load_default():
    """โหลด model อัตโนมัติตอน start — ลองทั้ง .pkl และ .h5"""
    candidates = [
        ("model_v1.h5",          "class_labels.json"),
        ("model_v1.pkl",         "class_labels.json"),
        ("thai_digit_model.h5",  "class_labels.json"),
        ("model_v1.joblib",      "class_labels.json"),
    ]
    for mpath, lpath in candidates:
        if os.path.exists(mpath) and os.path.exists(lpath):
            load_model_from_path(mpath, lpath)
            return
    print("[WARN] ไม่พบ model file — รัน: python train_local.py  แล้วมา restart app.py")


if __name__ == "__main__":
    try_load_default()
    app.run(host="0.0.0.0", port=5000, debug=True)
