import cv2

for i in range(3):
    print(f"🔹 Checking camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            print(f"✅ Camera {i} works! Press any key to close this preview window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"❌ Camera {i} not available")
