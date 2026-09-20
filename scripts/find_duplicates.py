import os
from PIL import Image
import imagehash

dataset_path =  r"./rayaru" # Change this

hashes = {}
duplicates = []

for cls in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, cls)

    if not os.path.isdir(class_path):
        continue

    print(f"\nChecking {cls}...")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            h = imagehash.average_hash(img)

            if h in hashes:
                duplicates.append((img_path, hashes[h]))
            else:
                hashes[h] = img_path

        except Exception:
            print("Error:", img_path)

print("\n--------------------------------")
print("Duplicate Images Found:", len(duplicates))

for d in duplicates:
    print("\nDuplicate:")
    print("Image 1:", d[0])
    print("Image 2:", d[1])