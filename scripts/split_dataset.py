import os
import shutil
from sklearn.model_selection import train_test_split

# ----------------------------
# Configuration
# ----------------------------
SOURCE_DIR = "rayaru"
OUTPUT_DIR = "dataset_final"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ----------------------------
# Create Output Folders
# ----------------------------
for split in ["train", "valid", "test"]:
    for cls in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, cls)

        if os.path.isdir(class_path):
            os.makedirs(
                os.path.join(OUTPUT_DIR, split, cls),
                exist_ok=True
            )

print("Folders Created Successfully!\n")

# ----------------------------
# Split Dataset
# ----------------------------
for cls in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, cls)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if len(images) < 5:
        print(f"Skipping {cls} (Too few images)")
        continue

    train_imgs, temp_imgs = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    valid_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "train", cls, img)
        )

    for img in valid_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "valid", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(OUTPUT_DIR, "test", cls, img)
        )

    print(f"{cls}")
    print(f"  Total : {len(images)}")
    print(f"  Train : {len(train_imgs)}")
    print(f"  Valid : {len(valid_imgs)}")
    print(f"  Test  : {len(test_imgs)}")
    print("-" * 35)

print("\nDataset Split Completed Successfully!")