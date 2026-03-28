# face_train.py
# Step 2 — Dataset se model train karo
# Run karo: python face_train.py

import cv2
import os
import json
import numpy as np

# ─────────────────────────────────────────────
#  SETTINGS  (datasetcreate.py ke saath match karo)
# ─────────────────────────────────────────────
DATASET_PATH = "dataset"
FACE_SIZE    = (100, 100)
# ─────────────────────────────────────────────


def main():
    print("=" * 40)
    print("  Face Model Training")
    print("=" * 40)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces     = []
    labels    = []
    label_map = {}    # { 0: "Rahul", 1: "Nisha", ... }

    # sorted() → alphabetical order → hamesha consistent labels
    person_names = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if not person_names:
        print(f"ERROR: '{DATASET_PATH}' folder mein koi person folder nahi mila!")
        print("Pehle datasetcreate.py chalao.")
        return

    print(f"\nPersons found: {person_names}\n")

    for current_label, person_name in enumerate(person_names):
        person_path = os.path.join(DATASET_PATH, person_name)
        label_map[current_label] = person_name

        img_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        loaded = 0
        for img_name in img_files:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  SKIP (load failed): {img_path}")
                continue

            # Preprocess — same as capture time
            img_eq      = cv2.equalizeHist(img)
            img_resized = cv2.resize(img_eq, FACE_SIZE, interpolation=cv2.INTER_CUBIC)

            faces.append(img_resized)
            labels.append(current_label)
            loaded += 1

        print(f"  Label {current_label:2d} → '{person_name}'  ({loaded} images loaded)")

        if loaded < 10:
            print(f"           ⚠ WARNING: Sirf {loaded} images hain."
                  "  Accuracy ke liye 30+ recommended hai.")

    print()

    if len(faces) == 0:
        print("ERROR: Koi image load nahi hui. Dataset check karo.")
        return

    # ── Train ───────────────────────────────────────────────────
    print(f"Training on {len(faces)} total face images...")
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.save("face_model.yml")
    print("✓ Model saved  → face_model.yml")

    # ── Label map JSON mein save karo ───────────────────────────
    label_map_str = {str(k): v for k, v in label_map.items()}
    with open("labels.json", "w") as f:
        json.dump(label_map_str, f, indent=2)
    print("✓ Labels saved → labels.json")

    print()
    print("=" * 40)
    print("  Training Complete!")
    print(f"  Persons: {list(label_map.values())}")
    print(f"  Total images: {len(faces)}")
    print("=" * 40)
    print("\nAb face_recognition.py chalao.")


if __name__ == "__main__":
    main()
