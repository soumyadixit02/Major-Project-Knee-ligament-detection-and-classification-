import os
import hashlib

# Path to your combined dataset
dataset_path = "./rayaru"

deleted = 0

for cls in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, cls)

    if not os.path.isdir(class_path):
        continue

    print(f"\nChecking {cls}...")

    hashes = {}

    for filename in os.listdir(class_path):

        file_path = os.path.join(class_path, filename)

        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            if file_hash in hashes:
                os.remove(file_path)
                deleted += 1
                print("Deleted:", filename)

            else:
                hashes[file_hash] = filename

        except Exception as e:
            print("Error:", filename, e)

print("\n========================")
print("Total Deleted:", deleted)
print("========================")