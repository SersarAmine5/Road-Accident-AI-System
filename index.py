import pdb

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class YOLOLightningModule(pl.LightningModule):
    def __init__(self, model_size, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        # Load with correct num_classes — this replaces the detection head
        yolo = YOLO(model_size)
        yolo.model.model[-1].nc = num_classes  # patch head class count
        # Rebuild the head with correct output channels
        self.model = yolo  # keep the full YOLO wrapper
        self.map_metric = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Use the YOLO wrapper's predict — returns Results objects, one per image
        results = self.model.predict(images, verbose=False)

        formatted_preds = []
        for r in results:
            boxes = r.boxes
            formatted_preds.append(
                {
                    "boxes": boxes.xyxy,  # [N, 4]
                    "scores": boxes.conf,  # [N]
                    "labels": boxes.cls.int(),  # [N]
                }
            )

        formatted_targets = [
            {"boxes": t["boxes"], "labels": t["labels"]} for t in targets
        ]
        self.map_metric.update(formatted_preds, formatted_targets)

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


class YOLODataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, img_size=640):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        self.train_ds = CocoDetection(
            root=f"{self.data_dir}/train",
            annFile=f"{self.data_dir}/train/_annotations.coco.json",
            transform=self.transform,
        )
        self.val_ds = CocoDetection(
            root=f"{self.data_dir}/valid",
            annFile=f"{self.data_dir}/valid/_annotations.coco.json",
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)


if __name__ == "__main__":
    data_module = YOLODataModule(data_dir="./data.coco", batch_size=16)

    model = YOLOLightningModule(
        model_size="yolov8n.pt",
        num_classes=3,
        lr=1e-3,
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val/mAP", mode="max", save_top_k=3),
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(monitor="val/mAP", patience=10, mode="max"),
        ],
    )

    trainer.fit(model, data_module)
