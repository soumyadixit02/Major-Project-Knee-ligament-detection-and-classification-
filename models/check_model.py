import torch

path = r"C:\Users\Sudarshan Systems\OneDrive\Desktop\knee_ligament_Project\models\best_model.pth"

data = torch.load(path, map_location="cpu")

print("Type:", type(data))

if isinstance(data, dict):
    print("\nNumber of saved items:", len(data))
    print("\nFirst 20 saved keys:")

    for i, key in enumerate(data.keys()):
        print(key)
        if i >= 19:
            break
else:
    print("The file is not a dictionary.")