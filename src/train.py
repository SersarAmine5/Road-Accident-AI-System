from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description="Simple file processor")

parser.add_argument("--epochs", default=10,type=int)
parser.add_argument("--data", default="./data/data.yaml" )
parser.add_argument("--model", default="./models/yolov8n.pt")

args = parser.parse_args()

model = YOLO(args.model)

# TODO: faire en sorte que les parameters d'entrainement soit modifiables.
# python src/train.py epochs=20

# pre-processing images (resize image, normliser contrast, etc.)
# entrainement
# evaluation
results = model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,
)
