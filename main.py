import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from deepface import DeepFace
from datetime import datetime
from deepface.basemodels import ArcFace
from numpy.linalg import norm

# === CONFIGURATION ===
DATASET_PATH = "raw_dataset"
OUTPUT_DIR = "working"
TEMP_FACE_DIR = "temp_faces"
EMBEDDINGS_FILE = "student_embeddings.pkl"
MODEL_NAME = "ArcFace"
INTERVAL_BETWEEN_CYCLES = 600   # seconds
FRAMES_PER_CYCLE = 4            # capture 4 frames before pause
# If you want to use laptop camera or inital camera connected to you system just set the CAMERA_URL = 0 (int form) remeber dont set it like CAMERA_URL = "0" (string form) 
CAMERA_URL = "yourlink provided by IP web cam app/"  # your phone camera stream URL 

# === CREATE REQUIRED FOLDERS ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

# === LOAD ARC FACE MODEL ===
print("🔄 Loading ArcFace model...")
arcface = ArcFace.load_model()
print("✅ ArcFace model loaded.")

# === LOAD STORED EMBEDDINGS ===
with open(EMBEDDINGS_FILE, "rb") as f:
    stored_embeddings = pickle.load(f)
print(f"✅ Loaded embeddings for {len(stored_embeddings)} students.")

# === UTILS ===
def l2_normalize(x):
    return x / (norm(x) + 1e-10)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)

# === FACE DETECTION WITH LIGHT & ROTATION FIX ===
detector = MTCNN()

def preprocess_face(face_img, keypoints=None, origin=None):
    """
    Safe preprocessing: lighting + optional rotation.
    - face_img: RGB cropped face image (numpy array)
    - keypoints: dict from MTCNN (with absolute coords if origin provided)
    - origin: (x1, y1) top-left of crop in original frame (so keypoints shift to local coords)
    """
    try:
        # --- Basic validation ---
        if face_img is None or face_img.size == 0:
            raise ValueError("Empty face_img")

        # --- Lighting Adjustment (Adaptive Gamma Correction) ---
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        mean = np.mean(gray)
        gamma = 1.0
        if mean < 100:
            gamma = 1.5
        elif mean > 160:
            gamma = 0.7
        look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
        face_img = cv2.LUT(face_img, look_up)

        # --- Rotation Correction (Align Eyes Horizontally) ---
        if keypoints and "left_eye" in keypoints and "right_eye" in keypoints:
            le = keypoints["left_eye"]
            re = keypoints["right_eye"]

            # Validate keypoints (must be two numeric values)
            def valid_kp(k):
                try:
                    if k is None: return False
                    if len(k) != 2: return False
                    x, y = float(k[0]), float(k[1])
                    if np.isnan(x) or np.isnan(y): return False
                    return True
                except Exception:
                    return False

            if valid_kp(le) and valid_kp(re):
                lx, ly = float(le[0]), float(le[1])
                rx, ry = float(re[0]), float(re[1])

                # Convert absolute -> local coords if origin provided
                if origin is not None:
                    ox, oy = origin
                    lx -= ox; rx -= ox
                    ly -= oy; ry -= oy

                # compute angle & center
                dx = rx - lx
                dy = ry - ly
                angle = np.degrees(np.arctan2(dy, dx))
                cx = (lx + rx) / 2.0
                cy = (ly + ry) / 2.0

                h, w = face_img.shape[0], face_img.shape[1]
                # clamp center inside image
                cx = max(0.0, min(float(w - 1), cx))
                cy = max(0.0, min(float(h - 1), cy))

                center = (float(cx), float(cy))
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                face_img = cv2.warpAffine(face_img, rot_matrix, (w, h),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                # invalid keypoints — skip rotation
                # (print once per call for debugging)
                print("⚠️ Eye keypoints invalid or NaN — skipping rotation.")
        # --- Histogram equalization for lighting uniformity ---
        try:
            img_yuv = cv2.cvtColor(face_img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            face_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        except Exception:
            pass

        return face_img

    except Exception as e:
        print(f"⚠️ Preprocessing failed: {e}")
        # return original (if face_img exists) or None to signal failure
        return face_img if (face_img is not None) else None

import uuid

def extract_faces_from_frame(frame, frame_idx):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    faces = []

    for i, res in enumerate(results):
        # MTCNN box can be negative; ensure ints and handle shapes
        x, y, w, h = res.get('box', (0,0,0,0))
        # make sure numbers are ints
        try:
            x = int(round(x)); y = int(round(y)); w = int(round(w)); h = int(round(h))
        except Exception:
            # skip malformed box
            continue

        padding = int(max(1, min(w, h) * 0.2))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_rgb.shape[1], x + w + padding)
        y2 = min(img_rgb.shape[0], y + h + padding)

        # validate coordinates
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid crop coords for face {i}: {(x1,y1,x2,y2)} — skipping")
            continue

        face = img_rgb[y1:y2, x1:x2]
        if face is None or face.size == 0:
            print(f"⚠️ Empty crop for face {i} — skipping")
            continue

        # Pass origin so preprocess can convert keypoints to face-local coords
        processed = preprocess_face(face, res.get('keypoints'), origin=(x1, y1))

        if processed is None or processed.size == 0:
            print(f"⚠️ Preprocessing returned empty for face {i} — skipping save")
            continue

        # unique filename with timestamp + uuid to avoid overwrite
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        face_filename = f"frame{frame_idx}_face{i}_{ts}_{uid}.jpg"
        face_path = os.path.join(TEMP_FACE_DIR, face_filename)

        # convert back to BGR for imwrite
        try:
            ok = cv2.imwrite(face_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            if ok:
                faces.append(face_path)
                print(f"💾 Saved face: {face_path}")
            else:
                print(f"❌ cv2.imwrite failed for {face_path}")
        except Exception as e:
            print(f"❌ Exception saving face {face_path}: {e}")

    return faces


# === EMBEDDING GENERATION ===
def get_embedding(face_path):
    try:
        rep = DeepFace.represent(
            img_path=face_path,
            model_name=MODEL_NAME,
            detector_backend="skip",  # already cropped
            enforce_detection=False
        )
        embedding = rep[0]["embedding"]
        return l2_normalize(np.array(embedding))
    except Exception as e:
        print(f"⚠️ Embedding failed for {face_path}: {e}")
        return None

# === FACE VERIFICATION ===
def verify_attendance(faces):
    present_students = set()

    for face_path in faces:
        face_emb = get_embedding(face_path)
        if face_emb is None:
            continue

        best_match = None
        best_score = -1

        # Compare with all embeddings of each student
        for student_id, emb_list in stored_embeddings.items():
            for ref_emb in emb_list:  # emb_list is a list of embeddings per student
                score = cosine_similarity(face_emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_match = student_id

        if best_score > 0.45:  # threshold (adjust between 0.4–0.55)
            present_students.add(best_match)
            print(f"✅ Match: {best_match} (similarity: {best_score:.4f})")
        else:
            print(f"❌ Unknown face (score: {best_score:.4f})")

    return list(present_students)

# === PROCESS EACH FRAME ===
def process_frame(frame, frame_idx):
    print("🔍 Extracting faces...")
    faces = extract_faces_from_frame(frame, frame_idx)
    print(f"🧑‍🤝‍🧑 Detected {len(faces)} face(s).")

    if not faces:
        print("⚠️ No faces found.")
        return

    print("🔎 Verifying faces...")
    present_students = verify_attendance(faces)

    if present_students:
        attendance = pd.DataFrame({
            "student_id": present_students,
            "status": "Present",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        attendance.to_csv(filepath, index=False)
        print(f"✅ Attendance saved: {filepath}")
    else:
        print("⚠️ No recognized students in this frame.")

# === MAIN LOOP ===
def main():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Error: Could not open camera stream.")
        return

    print("📷 Camera stream connected successfully.")
    print("🚀 Press Ctrl+C to stop the attendance loop.")

    try:
        while True:
            print(f"\n🕒 Starting new cycle at {datetime.now().strftime('%H:%M:%S')}...")
            for frame_idx in range(FRAMES_PER_CYCLE):
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to capture frame.")
                    continue

                process_frame(frame, frame_idx)
                time.sleep(2)  # small pause between frames

            print(f"⏸️ Cycle complete. Waiting {INTERVAL_BETWEEN_CYCLES/60:.1f} minutes...")
            time.sleep(INTERVAL_BETWEEN_CYCLES)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

