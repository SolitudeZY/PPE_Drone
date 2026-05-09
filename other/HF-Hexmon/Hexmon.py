# Hexmon/vyra-yolo-ppe-detection from hugging face
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Load from Hugging Face
# model = YOLO("hf://Hexmon/vyra-yolo-ppe-detection/best.pt")
model_path = hf_hub_download(repo_id="melihuzunoglu/ppe-detection", filename="best.pt", local_dir="/home/fs-ai/ppe_drone/other/HF-Hexmon/model")
model = YOLO(model_path)
# Run inference
results = model.predict("/home/fs-ai/ppe_drone/other/HF-Hexmon/test.png", conf=0.5)
results[0].show()
