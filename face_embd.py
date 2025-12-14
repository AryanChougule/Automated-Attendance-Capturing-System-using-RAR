import os
import pickle
import numpy as np
from deepface import DeepFace
from deepface.basemodels import ArcFace
from numpy.linalg import norm

# === CONFIG ===
DATASET_PATH = "raw_dataset"         # root folder containing student_id subfolders
EMBEDDINGS_FILE = "student_embeddings.pkl"
MODEL_NAME = "ArcFace"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# === UTIL ===
def l2_normalize(x):
    x = np.array(x, dtype=np.float32)
    return x / (norm(x) + 1e-10)

def is_image_file(fname):
    return os.path.splitext(fname.lower())[1] in ALLOWED_EXT

# === LOAD ARC FACE MODEL ===
print("🔄 Loading ArcFace model...")
model = ArcFace.load_model()
print("✅ ArcFace model loaded.")

# === BUILD EMBEDDINGS DB (list per student) ===
student_db = {}
total_images = 0
failed = 0

for student_id in os.listdir(DATASET_PATH):
    student_folder = os.path.join(DATASET_PATH, student_id)
    if not os.path.isdir(student_folder):
        continue

    embeddings_for_student = []
    print(f"\n📸 Processing student: {student_id}")

    for fname in sorted(os.listdir(student_folder)):
        if not is_image_file(fname):
            continue

        img_path = os.path.join(student_folder, fname)
        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                enforce_detection=False,   # allow still to get embedding if face is slightly off
                detector_backend="mtcnn"   # you can use "skip" if images are already tightly cropped
            )
            # DeepFace.represent returns a list of dicts; take first element's embedding
            emb = rep[0]["embedding"]
            emb = l2_normalize(emb)
            embeddings_for_student.append(emb)
            total_images += 1
            print(f"  ✅ {fname} -> embedding saved (total now {len(embeddings_for_student)})")
        except Exception as e:
            failed += 1
            print(f"  ⚠️ Failed {fname}: {e}")

    if embeddings_for_student:
        student_db[student_id] = embeddings_for_student
        print(f"  🧾 Stored {len(embeddings_for_student)} embeddings for '{student_id}'")
    else:
        print(f"  ⚠️ No valid images processed for '{student_id}'")

# === SAVE TO FILE ===
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(student_db, f)

print(f"\n✅ Done. Saved embeddings for {len(student_db)} students to '{EMBEDDINGS_FILE}'.")
print(f"Total images processed: {total_images}, failures: {failed}")
