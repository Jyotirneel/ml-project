import os
from src.inference import Predictor
from src.breed_info import BREED_INFO

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
model_path = os.path.join(project_root, "models", "cattle_breed_model.pt")
predictor = Predictor(model_path=model_path, breed_info=BREED_INFO)

# Samples folder
image_folder = os.path.join(project_root, "data", "samples")
os.makedirs(image_folder, exist_ok=True)

# Get all image files
image_files = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise FileNotFoundError("No images found in samples folder! Upload at least one image.")

# 🔹 Pick the newest image (creation time)
latest_image = max(image_files, key=os.path.getctime)

# 🔹 Delete all other images
for img in image_files:
    if img != latest_image:
        os.remove(img)

# Read the newest image
with open(latest_image, "rb") as f:
    image_bytes = f.read()

# Predict
results = predictor.predict(image_bytes)

# Print results
print(f"✅ Testing image: {latest_image}")
print("Predicted Breed Info:")
for k, v in results.items():
    print(f"{k.capitalize()}: {v}")
