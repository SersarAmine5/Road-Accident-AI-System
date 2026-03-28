import pdb

from PIL import Image
from ultralytics import YOLO

image = Image.open("./image.jpg")

model = YOLO("yolov8n.pt")

results = model.train(
    data="./data/data.yaml",
    epochs=10,
    imgsz=640,
)
