# datasetcreate.py
# Step 1 — Koi bhi apna naam type karo aur face capture karo
# Run karo: python datasetcreate.py

import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
DATASET_PATH  = "dataset"
FACE_SIZE     = (100, 100)   # Training ke saath same rehna chahiye
MAX_IMAGES    = 5           # Kitni images capture karni hain
AUTO_CAPTURE  = False         # True  → face detect hote hi automatically capture hoga
                             # False → SPACE dabao capture ke liye
# ─────────────────────────────────────────────


def get_name_from_gui():
    """Tkinter popup se naam lo."""
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)

    name = simpledialog.askstring(
        title="Face Dataset",
        prompt="Apna naam enter karo:",
        parent=root
    )
    root.destroy()

    if name:
        name = name.strip().title()   # "rahul" → "Rahul"
    return name


def main():
    # ── 1. Naam lo ──────────────────────────────────────────────
    person_name = get_name_from_gui()

    if not person_name:
        print("Naam nahi diya. Exiting.")
        return

    print(f"\nCapturing dataset for: {person_name}")

    # ── 2. Folder create karo ───────────────────────────────────
    person_path = os.path.join(DATASET_PATH, person_name)
    os.makedirs(person_path, exist_ok=True)

    existing = [f for f in os.listdir(person_path) if f.lower().endswith(".jpg")]
    count    = len(existing)
    print(f"Pehle se {count} images hain. {MAX_IMAGES - count} aur chahiye.")

    # ── 3. Haar Cascade load karo ───────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ── 4. Webcam ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam nahi mili!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    auto_counter  = 0
    STABLE_FRAMES = 8

    print("Instructions:")
    if AUTO_CAPTURE:
        print("  → Face saamne rakho — automatically capture hoga")
    else:
        print("  → SPACE dabao capture ke liye")
    print("  → Q dabao quit karne ke liye\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Detection ke liye sirf grayscale use karo ───────────
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq      = cv2.equalizeHist(gray)
        gray_blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray_blurred,
            scaleFactor  = 1.05,
            minNeighbors = 6,
            minSize      = (80, 80),
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        face_found = len(faces) > 0

        # ── UI ──────────────────────────────────────────────────
        display  = frame.copy()
        progress = int((count / MAX_IMAGES) * 640)
        cv2.rectangle(display, (0, 0), (progress, 8), (0, 255, 100), -1)

        cv2.putText(display, f"{person_name}  [{count}/{MAX_IMAGES}]",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if face_found:
            (x, y, w, h) = faces[0]

            if AUTO_CAPTURE:
                auto_counter += 1
                bar_fill = int((auto_counter / STABLE_FRAMES) * w)
                cv2.rectangle(display, (x, y + h + 4),
                              (x + bar_fill, y + h + 12), (0, 255, 0), -1)
                cv2.rectangle(display, (x, y + h + 4),
                              (x + w,       y + h + 12), (0, 255, 0), 1)
            else:
                auto_counter = 0

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── Capture ─────────────────────────────────────────
            should_capture = AUTO_CAPTURE and auto_counter >= STABLE_FRAMES

            if should_capture and count < MAX_IMAGES:
                auto_counter = 0

                # ✅ COLOR (BGR) frame se crop karo — RGB save hoga
                face_color   = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face_color, FACE_SIZE,
                                          interpolation=cv2.INTER_CUBIC)

                count    += 1
                save_path = os.path.join(person_path, f"{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                print(f"  Saved {count}/{MAX_IMAGES} — {save_path}")

                if count >= MAX_IMAGES:
                    cv2.putText(display, "DONE!", (220, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow("Dataset Capture", display)
                    cv2.waitKey(1500)
                    break
        else:
            auto_counter = 0
            cx, cy, bw, bh = 320, 240, 200, 200
            cv2.rectangle(display,
                          (cx - bw // 2, cy - bh // 2),
                          (cx + bw // 2, cy + bh // 2),
                          (0, 0, 255), 2)
            cv2.putText(display, "Face saamne laao",
                        (cx - 95, cy + bh // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Dataset Capture", display)
        key = cv2.waitKey(1) & 0xFF

        # Manual SPACE capture
        if key == ord(' ') and not AUTO_CAPTURE:
            if face_found and count < MAX_IMAGES:
                (x, y, w, h) = faces[0]

                # ✅ COLOR (BGR) frame se crop karo
                face_color   = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face_color, FACE_SIZE,
                                          interpolation=cv2.INTER_CUBIC)

                count    += 1
                save_path = os.path.join(person_path, f"{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                print(f"  Saved {count}/{MAX_IMAGES}")
            else:
                print("  Koi face detect nahi hua.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ {count} color images saved for '{person_name}' → {person_path}")
    print("Ab face_train.py chalao.")


if __name__ == "__main__":
    main()