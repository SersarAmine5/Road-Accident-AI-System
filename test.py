import pdb

from PIL import Image
from ultralytics import YOLO

image = Image.open("./image.jpg")

# model = YOLO("yolov8n.pt")

model = YOLO("/Users/nadir/amine-project/runs/detect/train7/weights/best.pt")

pdb.set_trace()

predictions = model.predict(image)

# results = model.train(
#     data="./data/data.yaml",
#     epochs=10,
#     imgsz=640,
# )
