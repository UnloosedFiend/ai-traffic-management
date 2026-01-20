import cv2
import os
import shutil
import random
from collections import defaultdict
from PIL import Image
import imagehash

# ================= CONFIG =================

VIDEO_ROOT = "videos"
IMG_OUT = "dataset/images"
LBL_OUT = "dataset/labels"

FRAME_SKIP = 7                 # extract 1 frame every N frames
PHASH_THRESHOLD = 8            # lower = stricter duplicate removal
TRAIN_SPLIT = 0.8

# =========================================

os.makedirs(f"{IMG_OUT}/train", exist_ok=True)
os.makedirs(f"{IMG_OUT}/val", exist_ok=True)
os.makedirs(f"{LBL_OUT}/train", exist_ok=True)
os.makedirs(f"{LBL_OUT}/val", exist_ok=True)


def is_duplicate(img_path, hash_list):
    img = Image.open(img_path)
    h = imagehash.phash(img)
    for prev in hash_list:
        if abs(h - prev) <= PHASH_THRESHOLD:
            return True
    hash_list.append(h)
    return False


all_images = defaultdict(list)

# ---------- STEP 1: EXTRACT FRAMES ----------
print("\n[STEP 1] Extracting frames from videos...")

for cam in sorted(os.listdir(VIDEO_ROOT)):
    cam_dir = os.path.join(VIDEO_ROOT, cam)
    if not os.path.isdir(cam_dir):
        continue

    print(f"\nCamera: {cam}")
    temp_dir = f"temp_frames/{cam}"
    os.makedirs(temp_dir, exist_ok=True)

    frame_counter = 0

    for video in sorted(os.listdir(cam_dir)):
        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        cap = cv2.VideoCapture(os.path.join(cam_dir, video))
        fid = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fid % FRAME_SKIP == 0:
                fname = f"{cam}_raw_{frame_counter:06d}.jpg"
                cv2.imwrite(os.path.join(temp_dir, fname), frame)
                frame_counter += 1

            fid += 1

        cap.release()

    print(f"  Extracted {frame_counter} raw frames")

# ---------- STEP 2: REMOVE DUPLICATES ----------
print("\n[STEP 2] Removing duplicate frames...")

for cam in sorted(os.listdir("temp_frames")):
    cam_temp = os.path.join("temp_frames", cam)
    hashes = []
    kept = []

    for img in sorted(os.listdir(cam_temp)):
        img_path = os.path.join(cam_temp, img)
        if not is_duplicate(img_path, hashes):
            kept.append(img_path)

    all_images[cam] = kept
    print(f"  {cam}: {len(kept)} unique frames kept")

# ---------- STEP 3: TRAIN / VAL SPLIT ----------
print("\n[STEP 3] Splitting into train / val...")

train_set = []
val_set = []

for cam, imgs in all_images.items():
    random.shuffle(imgs)
    split = int(len(imgs) * TRAIN_SPLIT)
    train_set.extend([(cam, p) for p in imgs[:split]])
    val_set.extend([(cam, p) for p in imgs[split:]])

# ---------- STEP 4: COPY + RENAME ----------
print("\n[STEP 4] Copying and renaming files...")

def copy_and_create_labels(dataset, split):
    counters = defaultdict(int)

    for cam, src in dataset:
        counters[cam] += 1
        name = f"{cam}_{counters[cam]:06d}.jpg"

        dst_img = os.path.join(IMG_OUT, split, name)
        dst_lbl = os.path.join(LBL_OUT, split, name.replace(".jpg", ".txt"))

        shutil.copy(src, dst_img)
        open(dst_lbl, "w").close()

copy_and_create_labels(train_set, "train")
copy_and_create_labels(val_set, "val")

# ---------- CLEANUP ----------
shutil.rmtree("temp_frames")

print("\n✅ DONE")
print(f"Train images: {len(train_set)}")
print(f"Val images:   {len(val_set)}")
