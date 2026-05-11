"""
Maximum Accuracy Training — ๒๑–๒๙
CNN บน CPU, dataset ลายมือจริง ~388 รูป/digit

รัน:  python train_local.py
ได้:  model_v1.h5  +  class_labels.json
"""
import os, sys, glob, random, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(
    SCRIPT_DIR,
    "thai-handwriting-number-master", "data", "raw",
    "thai-handwriting-number.appspot.com"
)
THAI_DIGITS       = {1:"๑",2:"๒",3:"๓",4:"๔",5:"๕",6:"๖",7:"๗",8:"๘",9:"๙"}
CLASSES           = [f"๒{THAI_DIGITS[d]}" for d in range(1, 10)]
IMG_SIZE          = 64
SAMPLES_PER_CLASS = 1000   # สูงสุดที่สมเหตุสมผล — 9,000 รูปรวม
EPOCHS            = 300
BATCH_SIZE        = 32
MODEL_FILE        = "model_v1.h5"
LABELS_FILE       = "class_labels.json"


# ── Preprocessing ─────────────────────────────────────────────────────────────
def center_digit(arr, padding_ratio=0.12):
    """
    หา bounding box ของตัวเลข → crop → center ใน canvas ใหม่
    ทำให้ทุกรูปมีตำแหน่งตัวเลขอยู่กลางเหมือนกัน
    """
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
    pad  = int(side * padding_ratio)
    side += pad * 2

    canvas = np.zeros((side, side), dtype=np.float32)
    y_off  = pad + (side - pad*2 - h) // 2
    x_off  = pad + (side - pad*2 - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop
    return canvas


def enhance_digit(arr):
    """เพิ่มความคมชัดของเส้น"""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1.2, 1.8))
    return np.array(img, dtype=np.float32)


def augment_digit(arr):
    """Augment รูปตัวเลขเดี่ยวก่อนนำมาต่อกัน"""
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # random rotation
    img = img.rotate(random.uniform(-15, 15), fillcolor=0)

    # random scale (zoom in/out)
    if random.random() > 0.4:
        scale = random.uniform(0.82, 1.15)
        new_size = int(img.width * scale)
        if new_size > 10:
            img = img.resize((new_size, new_size), Image.LANCZOS)
            canvas = Image.new("L", (arr.shape[1], arr.shape[0]), 0)
            x = (canvas.width  - img.width)  // 2
            y = (canvas.height - img.height) // 2
            canvas.paste(img, (max(0, x), max(0, y)))
            img = canvas

    # random blur หรือ sharpen
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
    elif random.random() > 0.7:
        img = img.filter(ImageFilter.SHARPEN)

    # random dilation/erosion (ทำให้เส้นหนา/บางขึ้น)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.MaxFilter(3))
    elif random.random() > 0.6:
        img = img.filter(ImageFilter.MinFilter(3))

    return np.array(img, dtype=np.float32)


# ── Load raw single-digit images ──────────────────────────────────────────────
def load_folder(folder_num):
    folder_path = os.path.join(DATASET_ROOT, str(folder_num))
    images = []
    for fpath in glob.glob(os.path.join(folder_path, "*.png")):
        img = Image.open(fpath).convert("RGBA")
        bg  = Image.new("L", img.size, 255)
        bg.paste(img.convert("L"))
        arr = 255.0 - np.array(bg, dtype=np.float32)   # invert → พื้นดำ ตัวขาว
        arr = center_digit(arr)                         # center ตัวเลข
        arr = enhance_digit(arr)                        # เพิ่มความคมชัด
        if arr.mean() > 2:                              # กรองรูปว่าง
            images.append(arr)
    return images


def make_combined(img2_arr, imgX_arr):
    """
    Augment แต่ละหลัก → resize เป็น 32×64 → ต่อแนวนอน → 64×64
    ใส่ noise เล็กน้อยในภาพรวม
    """
    half = IMG_SIZE // 2
    arr2 = augment_digit(img2_arr.copy())
    arrX = augment_digit(imgX_arr.copy())

    pil2 = Image.fromarray(np.clip(arr2, 0, 255).astype(np.uint8)).resize((half, IMG_SIZE), Image.LANCZOS)
    pilX = Image.fromarray(np.clip(arrX, 0, 255).astype(np.uint8)).resize((half, IMG_SIZE), Image.LANCZOS)

    combined = np.concatenate(
        [np.array(pil2, dtype=np.float32), np.array(pilX, dtype=np.float32)],
        axis=1
    )
    if random.random() > 0.35:
        combined = np.clip(combined + np.random.normal(0, random.uniform(2, 10), combined.shape), 0, 255)

    return combined


# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MAX ACCURACY CNN — ๒๑–๒๙  (CPU Training)")
print("=" * 60)

if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"ไม่พบ dataset:\n  {DATASET_ROOT}")

print("\n[1/5] โหลด dataset...")
t0 = time.time()
digit2_imgs = load_folder(2)
digit_imgs  = {d: load_folder(d) for d in range(1, 10)}
for d in [2] + list(range(1, 10)):
    n = len(digit2_imgs) if d == 2 else len(digit_imgs[d])
    print(f"  folder {d} ({THAI_DIGITS[d]}): {n} รูป (หลังกรอง)")
print(f"  โหลดเสร็จใน {time.time()-t0:.1f}s")

print(f"\n[2/5] สร้างรูปสองหลัก ({SAMPLES_PER_CLASS}/class, รวม {SAMPLES_PER_CLASS*9:,} รูป)...")
t0 = time.time()
X, y = [], []
for label_idx, d in enumerate(range(1, 10)):
    pool2 = digit2_imgs
    poolX = digit_imgs[d]
    for _ in range(SAMPLES_PER_CLASS):
        combined = make_combined(random.choice(pool2), random.choice(poolX))
        X.append(combined)
        y.append(label_idx)
    print(f"  {CLASSES[label_idx]}: {SAMPLES_PER_CLASS} samples ✓")

X = np.array(X, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)
print(f"  Dataset shape: {X.shape}  สร้างเสร็จใน {time.time()-t0:.1f}s")

print("\n[3/5] แบ่ง Train/Test (80/20)...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

import keras
from keras import layers

y_train_cat = keras.utils.to_categorical(y_train, len(CLASSES))
y_test_cat  = keras.utils.to_categorical(y_test,  len(CLASSES))

print("\n[4/5] Build & Train CNN...")

# ── Model Architecture ────────────────────────────────────────────────────────
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(CLASSES)):
    inp = keras.Input(shape=input_shape)

    # Augmentation layers (active only during training)
    x = layers.RandomRotation(0.033)(inp)           # ±12 degrees
    x = layers.RandomTranslation(0.10, 0.10)(x)    # ±10%
    x = layers.RandomZoom(0.12)(x)                 # ±12%

    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4 — deep features
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Classifier
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"]
    )
    return model


model = build_model()
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=8,
        min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True,
        monitor="val_accuracy", verbose=1
    ),
]

t0 = time.time()
history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)
elapsed = time.time() - t0
print(f"\n  Train เสร็จใน {elapsed/60:.1f} นาที")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n[5/5] Evaluate & Save...")

if os.path.exists("best_model.h5"):
    model = keras.models.load_model("best_model.h5")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"  Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print(f"{'='*60}\n")
print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print("       " + "  ".join(f"{c:>5}" for c in CLASSES))
for i, row in enumerate(cm):
    correct = "✓" if row[i] == row.max() else " "
    print(f"  {CLASSES[i]:>3}  " + "  ".join(f"{v:>5}" for v in row) + f"  {correct}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_FILE)
size_kb = os.path.getsize(MODEL_FILE) / 1024
print(f"\n  Saved: {MODEL_FILE}  ({size_kb:.0f} KB)")

meta = {"classes": CLASSES, "img_size": IMG_SIZE, "model_type": "keras"}
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"  Saved: {LABELS_FILE}")

# แสดง training curve สั้นๆ
best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
best_val   = max(history.history["val_accuracy"])
print(f"\n  Best epoch   : {best_epoch}")
print(f"  Best val_acc : {best_val:.4f}  ({best_val*100:.2f}%)")
print(f"  Final test   : {acc:.4f}  ({acc*100:.2f}%)")

print(f"\n{'='*60}")
print(f"  เสร็จ! วาง {MODEL_FILE} + {LABELS_FILE} ไว้ใน folder เดียวกัน")
print(f"  กับ app.py แล้วรัน:  python app.py")
print(f"  เปิดเว็บ: http://localhost:5000")
print(f"{'='*60}")
