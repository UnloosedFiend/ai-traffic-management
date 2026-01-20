import os
import hashlib
from PIL import Image
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim

FRAMES_ROOT = "frames"

PHASH_THRESHOLD = 5
SSIM_THRESHOLD = 0.95
KEEP_EVERY_N = 3

EXT = (".jpg", ".jpeg", ".png")

for cam in sorted(os.listdir(FRAMES_ROOT)):
    cam_path = os.path.join(FRAMES_ROOT, cam)

    if not os.path.isdir(cam_path):
        continue

    print(f"\nProcessing {cam}")

    files = sorted([f for f in os.listdir(cam_path) if f.lower().endswith(EXT)])

    # ---- STEP 1: exact duplicates ----
    seen = set()
    step1 = []

    for f in files:
        p = os.path.join(cam_path, f)
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()

        if h not in seen:
            seen.add(h)
            step1.append(f)
        else:
            os.remove(p)

    # ---- STEP 2: pHash still-frame removal ----
    step2 = []
    prev_hash = None

    for f in step1:
        p = os.path.join(cam_path, f)
        h = imagehash.phash(Image.open(p))

        if prev_hash is not None and (h - prev_hash) <= PHASH_THRESHOLD:
            os.remove(p)
        else:
            step2.append(f)
            prev_hash = h

    # ---- STEP 3: SSIM slow-motion filter ----
    step3 = []
    prev_img = None

    for f in step2:
        p = os.path.join(cam_path, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        if prev_img is not None:
            score, _ = ssim(prev_img, img, full=True)
            if score > SSIM_THRESHOLD:
                os.remove(p)
                continue

        step3.append(f)
        prev_img = img

    # ---- STEP 4: dataset balancing ----
    for i, f in enumerate(step3):
        if i % KEEP_EVERY_N != 0:
            os.remove(os.path.join(cam_path, f))

    remaining = len([f for f in os.listdir(cam_path) if f.lower().endswith(EXT)])
    print(f"[DONE] {cam}: {remaining} frames kept")
