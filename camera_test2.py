import cv2
import os
from datetime import datetime

# === TRY MULTIPLE CAMERA INDICES FOR DROIDCAM ===
def get_droidcam():
    print("🔍 Searching for DroidCam virtual camera...")

    for idx in range(0, 6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"✅ Found camera at index {idx}")
            return cap
    return None

SAVE_DIR = "temp_faces"
CAPTURE_INTERVAL = 10  # seconds between saved frames

os.makedirs(SAVE_DIR, exist_ok=True)

cap = get_droidcam()

if cap is None:
    print("❌ Could not find DroidCam virtual camera.")
    print("➡ Make sure:")
    print("   - DroidCam Windows client is running")
    print("   - Phone is connected")
    print("   - Video is enabled")
    exit()

print("📸 Press 'q' to stop capturing.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        continue

    cv2.imshow("DroidCam Feed", frame)

    filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_count}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"💾 Saved: {filepath}")
    frame_count += 1

    key = cv2.waitKey(CAPTURE_INTERVAL * 1000) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
