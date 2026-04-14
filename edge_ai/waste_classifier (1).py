"""
========================================================
  Wet / Dry Waste Classifier — RPi-Ready Training Script
  UPDATED for nested subfolder structure:

  Dataset/
    train/
      dry/
        leaf_waste/    *.jpg
        paper_waste/   *.jpg
        ... (more subfolders)
      wet/
        food_waste/    *.jpg
        ... (more subfolders)
    val/
      dry/ ...
      wet/ ...
    test/              ← optional
      dry/ ...
      wet/ ...

  Label is assigned from the TOP-LEVEL folder (dry / wet).
  Subfolders are ignored for labelling — every image inside
  dry/**  is "dry", every image inside wet/**  is "wet".
========================================================
"""

# ──────────────────────────────────────────────────────────
# 0. Imports & Config
# ──────────────────────────────────────────────────────────
import os, gc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
from PIL import Image, UnidentifiedImageError

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Paths — update DATASET_DIR to your folder ─────────────
DATASET_DIR = r"C:\Users\HP\Downloads\Dataset"   # ← your path
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val")
TEST_DIR    = os.path.join(DATASET_DIR, "test")   # optional
SAVE_DIR    = os.path.join(DATASET_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

KERAS_PATH  = os.path.join(SAVE_DIR, "waste_classifier.keras")
TFLITE_INT8 = os.path.join(SAVE_DIR, "waste_classifier_quantised.tflite")
TFLITE_F16  = os.path.join(SAVE_DIR, "waste_classifier_f16.tflite")

# ── Hyper-parameters ──────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16           # reduce to 8 if you hit OOM
EPOCHS_1    = 10
EPOCHS_2    = 10
LR_HEAD     = 1e-3
LR_FINETUNE = 1e-5

# ↓ Must match your top-level folder names exactly (case-sensitive on Linux)
CLASSES     = ["dry", "wet"]
NUM_CLASSES = len(CLASSES)
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print(f"TensorFlow  : {tf.__version__}")
print(f"GPUs visible: {tf.config.list_physical_devices('GPU')}")


# ──────────────────────────────────────────────────────────
# 1. Recursive File Collector  ← KEY CHANGE
#
#    Old code used flow_from_directory which only looks ONE
#    level deep:  train/dry/*.jpg  ✓  train/dry/leaf_waste/*.jpg  ✗
#
#    New code uses Path.rglob("*") to walk ALL subfolders and
#    assigns the label based on the TOP-LEVEL class folder.
# ──────────────────────────────────────────────────────────
def collect_files(split_dir: str, classes: list) -> tuple:
    """
    Recursively collects every image under each class folder.

    Example:
      collect_files("dataset/train", ["dry", "wet"])
      → scans  dataset/train/dry/**  (label 0)
               dataset/train/wet/**  (label 1)
      → returns ([path1, path2, ...], [0, 0, 1, 1, ...])
    """
    file_paths, labels = [], []

    for label_idx, cls in enumerate(classes):
        cls_root = Path(split_dir) / cls

        if not cls_root.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Class folder not found: {cls_root}"
                f"\nExpected folders inside '{split_dir}': {classes}"
            )

        # rglob("*") walks every nested subfolder automatically
        found = [
            str(p) for p in cls_root.rglob("*")
            if p.suffix.lower() in IMG_EXTS and p.is_file()
        ]

        if not found:
            print(f"  ⚠  WARNING: No images found under {cls_root}")

        file_paths.extend(found)
        labels.extend([label_idx] * len(found))

        # Show breakdown by subfolder so you can verify counts
        subfolders = sorted({str(Path(p).parent) for p in found})
        print(f"  [{cls}]  {len(found)} images across {len(subfolders)} subfolder(s):")
        for sf in subfolders:
            sf_count = sum(1 for p in found if str(Path(p).parent) == sf)
            print(f"           {sf_count:>5}  {sf}")

    print(f"\n  TOTAL {split_dir}: {len(file_paths)} images\n")
    return file_paths, labels


def verify_images(file_paths: list, labels: list) -> tuple:
    """
    Opens every image with Pillow to detect corrupt/truncated files.
    Removes bad files from the list before they can crash tf.data.

    Common causes caught here:
      - Corrupted JPEG ("Improper call to JPEG library")
      - Truncated PNG
      - Zero-byte files
      - Wrong extension (e.g. a .jpg that is actually a text file)
    """
    good_paths, good_labels = [], []
    bad_count = 0

    print(f"  Verifying {len(file_paths)} images (this runs once)...")
    for path, label in zip(file_paths, labels):
        try:
            with Image.open(path) as img:
                img.verify()           # catches structural corruption
            # verify() leaves the file in an unusable state — reopen to
            # confirm pixel data can actually be decoded (catches truncation)
            with Image.open(path) as img:
                img.load()
            good_paths.append(path)
            good_labels.append(label)
        except (UnidentifiedImageError, OSError, SyntaxError, Exception):
            print(f"  ✗ Skipping corrupt file: {path}")
            bad_count += 1

    print(f"  Removed {bad_count} corrupt file(s). "
          f"{len(good_paths)} clean images remain.\n")
    return good_paths, good_labels


print("── Scanning TRAIN ──")
train_paths, train_labels = collect_files(TRAIN_DIR, CLASSES)
train_paths, train_labels = verify_images(train_paths, train_labels)

print("── Scanning VAL ──")
val_paths, val_labels = collect_files(VAL_DIR, CLASSES)
val_paths, val_labels = verify_images(val_paths, val_labels)

test_paths, test_labels = [], []
if os.path.isdir(TEST_DIR):
    print("── Scanning TEST ──")
    test_paths, test_labels = collect_files(TEST_DIR, CLASSES)
    test_paths, test_labels = verify_images(test_paths, test_labels)


# ──────────────────────────────────────────────────────────
# 2. tf.data Pipeline
#    Replaces ImageDataGenerator — streams images batch-by-batch
#    so the full dataset never loads into RAM at once.
# ──────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path: tf.Tensor, label: tf.Tensor,
               augment: bool = False) -> tuple:
    raw = tf.io.read_file(path)

    # decode_image auto-detects the actual file format (JPEG, PNG, BMP, GIF)
    # from the file header — not the extension. This safely handles files
    # whose extension doesn't match their real format (e.g. BMP saved as .jpg).
    image = tf.image.decode_image(
        raw, channels=3,
        expand_animations=False,
        dtype=tf.uint8,
    )
    image.set_shape([None, None, 3])   # required for resize after decode_image

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)

    return image, label


def make_dataset(paths: list, labels: list,
                 augment: bool = False,
                 shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=SEED)

    ds = ds.map(
        lambda p, l: load_image(p, l, augment=augment),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


train_ds = make_dataset(train_paths, train_labels, augment=True,  shuffle=True)
val_ds   = make_dataset(val_paths,   val_labels,   augment=False, shuffle=False)
test_ds  = (make_dataset(test_paths, test_labels, augment=False, shuffle=False)
            if test_paths else None)

# Extra augmentation layers (rotation/zoom handled inside model)
# augment_layer = tf.keras.Sequential([
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomTranslation(0.1, 0.1),
# ], name="augmentation")


# ──────────────────────────────────────────────────────────
# 3. Build Model
# ──────────────────────────────────────────────────────────
def build_model(num_classes: int, trainable_base: bool = False) -> tf.keras.Model:
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = inputs # x = augment_layer(inputs, training=True)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = (
        layers.Dense(1, activation="sigmoid")(x)
        if num_classes == 2 else
        layers.Dense(num_classes, activation="softmax")(x)
    )
    return tf.keras.Model(inputs, outputs)


model = build_model(NUM_CLASSES, trainable_base=False)
model.summary(line_length=90)

loss_fn = "binary_crossentropy" if NUM_CLASSES == 2 else "sparse_categorical_crossentropy"


# ──────────────────────────────────────────────────────────
# 4. Phase 1 — Train Head Only
# ──────────────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(LR_HEAD),
    loss=loss_fn,
    metrics=["accuracy"],
)

cb_list = [
    callbacks.ModelCheckpoint(
        KERAS_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    ),
]

print("\n── Phase 1: Training classification head ──")
history1 = model.fit(
    train_ds,
    epochs=EPOCHS_1,
    validation_data=val_ds,
    callbacks=cb_list,
    verbose=1,
)
gc.collect()


# ──────────────────────────────────────────────────────────
# 5. Phase 2 — Fine-tune Last 30 Layers
# ──────────────────────────────────────────────────────────
print("\n── Phase 2: Fine-tuning base model ──")
base_model = model.get_layer("mobilenetv2_1.00_224")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(LR_FINETUNE),
    loss=loss_fn,
    metrics=["accuracy"],
)

history2 = model.fit(
    train_ds,
    epochs=EPOCHS_2,
    validation_data=val_ds,
    callbacks=cb_list,
    verbose=1,
)
gc.collect()


# ──────────────────────────────────────────────────────────
# 6. Evaluation
# ──────────────────────────────────────────────────────────
best_model = tf.keras.models.load_model(KERAS_PATH)
val_loss, val_acc = best_model.evaluate(val_ds, verbose=1)
print(f"\nVal Loss : {val_loss:.4f}")
print(f"Val Acc  : {val_acc*100:.2f}%")

if test_ds:
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc*100:.2f}%")


# ──────────────────────────────────────────────────────────
# 7. Training Curves
# ──────────────────────────────────────────────────────────
acc   = history1.history["accuracy"]      + history2.history["accuracy"]
vacc  = history1.history["val_accuracy"]  + history2.history["val_accuracy"]
loss  = history1.history["loss"]          + history2.history["loss"]
vloss = history1.history["val_loss"]      + history2.history["val_loss"]
split = len(history1.history["accuracy"])
eps   = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
ax1.plot(eps, acc, label="Train"); ax1.plot(eps, vacc, label="Val")
ax1.axvline(split, color="gray", linestyle="--", label="Fine-tune start")
ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True)

ax2.plot(eps, loss, label="Train"); ax2.plot(eps, vloss, label="Val")
ax2.axvline(split, color="gray", linestyle="--")
ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
plt.show()


# ──────────────────────────────────────────────────────────
# 8. TFLite Export  ← upload this to Raspberry Pi
# ──────────────────────────────────────────────────────────
print("\n── Converting to TFLite INT8 ──")

def representative_dataset():
    for i, (images, _) in enumerate(train_ds):
        yield [images.numpy().astype(np.float32)]
        if i >= 100:
            break

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8 = converter.convert()
with open(TFLITE_INT8, "wb") as f:
    f.write(tflite_int8)
print(f"INT8 TFLite → {TFLITE_INT8}  ({os.path.getsize(TFLITE_INT8)/1e6:.1f} MB)")

print("\n── Converting to TFLite FP16 ──")
conv_f16 = tf.lite.TFLiteConverter.from_keras_model(best_model)
conv_f16.optimizations = [tf.lite.Optimize.DEFAULT]
conv_f16.target_spec.supported_types = [tf.float16]
tflite_f16 = conv_f16.convert()
with open(TFLITE_F16, "wb") as f:
    f.write(tflite_f16)
print(f"FP16 TFLite → {TFLITE_F16}  ({os.path.getsize(TFLITE_F16)/1e6:.1f} MB)")

print("\n" + "="*60)
print("  DONE — upload the .tflite file to your Raspberry Pi")
print("="*60)