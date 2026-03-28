from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# TODO: faire en sorte que les parameters d'entrainement soit modifiables.
# python src/train.py epochs=20

# pre-processing images (resize image, normliser contrast, etc.)
# entrainement
# evaluation
results = model.train(
    data="./data/data.yaml",
    epochs=10,
    imgsz=640,
)
