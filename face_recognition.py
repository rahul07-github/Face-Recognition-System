# face_recognition.py
# Step 3 — Live face pehchano aur naam dikhao
# Run karo: python face_recognition.py

import cv2
import json
import numpy as np

# ─────────────────────────────────────────────
#  SETTINGS  (datasetcreate.py ke saath match karo)
# ─────────────────────────────────────────────
FACE_SIZE            = (100, 100)
CONFIDENCE_THRESHOLD = 90    # LBPH mein: lower = better match
                             # 70 ek acchi starting value hai
                             # agar galat naam aa raha ho → 60 try karo
                             # agar known faces bhi Unknown dikh rahe → 80 karo
# ─────────────────────────────────────────────


def load_resources():
    """Model aur labels load karo."""

    # ── Labels ──────────────────────────────────────────────────
    try:
        with open("labels.json", "r") as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        print(f"✓ Labels loaded: {label_map}")
    except FileNotFoundError:
        print("ERROR: labels.json nahi mili!")
        print("Pehle face_train.py chalao.")
        return None, None

    # ── Model ───────────────────────────────────────────────────
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read("face_model.yml")
        print("✓ Model loaded: face_model.yml")
    except Exception as e:
        print(f"ERROR: face_model.yml load nahi hua — {e}")
        print("Pehle face_train.py chalao.")
        return None, None

    return face_recognizer, label_map


def main():
    print("=" * 40)
    print("  Face Recognition")
    print("=" * 40)

    face_recognizer, label_map = load_resources()
    if face_recognizer is None:
        return

    # Haar Cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam nahi mili!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    print(f"\nConfidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Same preprocessing as training time ─────────────────
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq  = cv2.equalizeHist(gray)
        gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray_blur,
            scaleFactor  = 1.05,
            minNeighbors = 6,
            minSize      = (80, 80),
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Face crop karo aur same preprocessing lagao
            face_crop    = gray[y:y + h, x:x + w]
            face_eq      = cv2.equalizeHist(face_crop)
            face_resized = cv2.resize(face_eq, FACE_SIZE,
                                       interpolation=cv2.INTER_CUBIC)

            # Predict
            label, confidence = face_recognizer.predict(face_resized)

            # ── Confidence decide karega naam ya "Unknown" ───────
            if confidence < CONFIDENCE_THRESHOLD:
                name  = label_map.get(label, "Unknown")
                color = (0, 220, 0)       # Green — pehchana
                conf_bar_color = (0, 220, 0)
            else:
                name  = "Unknown"
                color = (0, 0, 220)       # Red — nahi pehchana
                conf_bar_color = (0, 0, 220)

            # ── Confidence bar draw karo ─────────────────────────
            # confidence 0 (best) se ~150 (worst) ke beech hoti hai
            # Bar fill: 0 conf → full green, 100+ → mostly red
            bar_max   = 120
            bar_fill  = max(0, int((1 - confidence / bar_max) * w))
            cv2.rectangle(frame, (x, y + h + 4),
                          (x + bar_fill, y + h + 14), conf_bar_color, -1)
            cv2.rectangle(frame, (x, y + h + 4),
                          (x + w, y + h + 14), (180, 180, 180), 1)

            # ── Name + confidence text ───────────────────────────
            label_text = f"{name}  ({confidence:.1f})"
            cv2.putText(frame, label_text, (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ── Face box ─────────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # ── Instructions ────────────────────────────────────────
        cv2.putText(frame, "Press Q to quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition band ho gayi.")


if __name__ == "__main__":
    main()
