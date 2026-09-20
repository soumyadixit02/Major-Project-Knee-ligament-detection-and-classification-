import os

dataset_path = "dataset_final"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for split in ["train", "valid", "test"]:
    split_path = os.path.join(dataset_path, split)

    print("\n" + "=" * 40)
    print(split.upper())
    print("=" * 40)

    if not os.path.exists(split_path):
        print("Folder not found:", split_path)
        continue

    total = 0

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_path):
            count = sum(
                1 for file in os.listdir(class_path)
                if file.lower().endswith(image_extensions)
            )

            print(f"{class_name}: {count}")
            total += count

    print("TOTAL:", total)