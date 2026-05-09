from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(repo_id="melihuzunoglu/ppe-detection", filename="best.pt", local_dir="/home/fs-ai/ppe_drone/other/HF-ppe/model")
model = YOLO(model_path)
source = "http://images.cocodataset.org/val2017/000000039769.jpg"
model.predict(source=source, save=True)
