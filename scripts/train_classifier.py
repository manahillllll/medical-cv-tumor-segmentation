"""
Train the classification head on top of a pretrained segmentation encoder.

Per milestone 5: trains with the encoder frozen for `freeze_encoder_epochs`, then
unfreezes for end-to-end fine-tuning. Expects a CSV mapping case_id -> class label
(see scripts/download_data.md for where BraTS grade/subtype labels come from).

Usage:
    python scripts/train_classifier.py --config config.yaml --labels_csv data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from monai.data import Dataset, list_data_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import discover_cases, get_classification_val_transforms, get_train_transforms
from src.models.classifier import SegClassifier
from src.models.unet3d import UNet3D
from src.utils import load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--labels_csv", type=str, required=True,
                    help="CSV with columns: case_id,label (integer class index)")
    return p.parse_args()


def load_labels(csv_path: str) -> dict[str, int]:
    labels = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["case_id"]] = int(row["label"])
    return labels


def build_classification_dicts(data_dir: str, labels: dict[str, int]) -> list[dict]:
    cases = discover_cases(data_dir, require_label=False)
    dicts = []
    for c in cases:
        if c.case_id not in labels:
            continue
        dicts.append({"image": c.images, "label": c.label, "case_id": c.case_id,
                       "cls_label": labels[c.case_id]})
    return dicts


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train_classifier"]
    patch_size = tuple(dcfg["patch_size"])
    labels = load_labels(args.labels_csv)
    data_dicts = build_classification_dicts(dcfg["data_dir"], labels)

    n_val = max(1, int(len(data_dicts) * dcfg["val_fraction"]))
    val_dicts, train_dicts = data_dicts[:n_val], data_dicts[n_val:]

    train_ds = Dataset(data=train_dicts, transform=get_train_transforms(patch_size))
    val_ds = Dataset(data=val_dicts, transform=get_classification_val_transforms(patch_size))
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=dcfg["num_workers"], collate_fn=list_data_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                             num_workers=dcfg["num_workers"], collate_fn=list_data_collate)

    unet = UNet3D(in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
                   base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"])
    load_checkpoint(tcfg["seg_checkpoint"], unet, map_location=device)

    model = SegClassifier(unet, num_classification_classes=mcfg["num_cls_classes"],
                           dropout_p=mcfg["cls_dropout_p"], freeze_encoder=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(tcfg["log_dir"])

    best_acc = 0.0
    global_step = 0
    for epoch in range(tcfg["num_epochs"]):
        model.freeze_encoder = epoch < tcfg["freeze_encoder_epochs"]
        model.train()

        pbar = tqdm(train_loader, desc=f"epoch {epoch} (encoder {'frozen' if model.freeze_encoder else 'trainable'})")
        for batch in pbar:
            images = batch["image"].to(device)
            cls_labels = torch.as_tensor(batch["cls_label"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            _, cls_logits = model(images)
            loss = criterion(cls_logits, cls_labels)
            loss.backward()
            optimizer.step()

            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                cls_label = torch.as_tensor(batch["cls_label"]).to(device)
                _, cls_logits = model(images)
                pred = cls_logits.argmax(dim=1)
                correct += (pred == cls_label).sum().item()
                total += cls_label.numel()

        acc = correct / max(total, 1)
        writer.add_scalar("val/accuracy", acc, epoch)
        print(f"epoch {epoch}: val accuracy {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(Path(tcfg["checkpoint_dir"]) / "best.pt", model, optimizer, epoch,
                             extra={"val_accuracy": acc})

        save_checkpoint(Path(tcfg["checkpoint_dir"]) / "last.pt", model, optimizer, epoch)

    writer.close()


if __name__ == "__main__":
    main()
