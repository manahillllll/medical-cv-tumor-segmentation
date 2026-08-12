"""
Train the 3D U-Net segmentation model.

Milestone 2 sanity check: run with train_segmentation.overfit_single_batch: true in
config.yaml to confirm the model can drive loss to ~0 on a single batch before
committing to a full training run.

Usage:
    python scripts/train_segmentation.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from monai.data import list_data_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_datasets
from src.losses import DiceCELoss, compute_class_weights
from src.metrics import brats_region_dice
from src.models.unet3d import UNet3D
from src.utils import save_checkpoint, set_seed, to_plain_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg, mcfg, lcfg, tcfg = cfg["data"], cfg["model"], cfg["loss"], cfg["train_segmentation"]
    patch_size = tuple(dcfg["patch_size"])

    train_ds, val_ds = build_datasets(
        dcfg["data_dir"], patch_size=patch_size, val_fraction=dcfg["val_fraction"], cache=dcfg["cache"],
        format=dcfg.get("format", "nifti"),
    )
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                               num_workers=dcfg["num_workers"], drop_last=True,
                               collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = UNet3D(
        in_channels=mcfg["in_channels"], num_classes=mcfg["num_seg_classes"],
        base_filters=mcfg["base_filters"], depth=mcfg["depth"], dropout_p=mcfg["dropout_p"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["num_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"])

    criterion = DiceCELoss(
        num_classes=mcfg["num_seg_classes"], weight_dice=lcfg["weight_dice"], weight_ce=lcfg["weight_ce"],
        include_background_in_dice=lcfg["include_background_in_dice"],
    )

    writer = SummaryWriter(tcfg["log_dir"])
    best_dice = 0.0

    overfit_batch = None
    if tcfg["overfit_single_batch"]:
        overfit_batch = next(iter(train_loader))
        print("Overfit-single-batch sanity check enabled.")

    global_step = 0
    for epoch in range(tcfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        batches = [overfit_batch] if overfit_batch is not None else train_loader
        n_batches = 1 if overfit_batch is not None else len(train_loader)

        pbar = tqdm(batches, total=n_batches, desc=f"epoch {epoch}")
        for batch in pbar:
            images = to_plain_tensor(batch["image"]).to(device)
            labels = to_plain_tensor(batch["label"]).to(device).squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tcfg["amp"]):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"epoch {epoch}: mean train loss {epoch_loss / n_batches:.4f}")

        if (epoch + 1) % tcfg["val_every"] == 0 and overfit_batch is None:
            model.eval()
            region_dices = {"whole_tumor": [], "tumor_core": [], "enhancing_tumor": []}
            with torch.no_grad():
                for val_batch in val_loader:
                    image = to_plain_tensor(val_batch["image"]).to(device)
                    label = to_plain_tensor(val_batch["label"]).to(device).squeeze(1).squeeze(0)
                    logits = model(image)
                    pred = logits.argmax(dim=1).squeeze(0)
                    regions = brats_region_dice(pred.cpu(), label.cpu())
                    for k, v in regions.items():
                        region_dices[k].append(v)

            mean_dices = {k: sum(v) / max(len(v), 1) for k, v in region_dices.items()}
            mean_overall = sum(mean_dices.values()) / len(mean_dices)
            for k, v in mean_dices.items():
                writer.add_scalar(f"val/dice_{k}", v, epoch)
            print(f"epoch {epoch}: val dice {mean_dices}")

            if mean_overall > best_dice:
                best_dice = mean_overall
                save_checkpoint(Path(tcfg["checkpoint_dir"]) / "best.pt", model, optimizer, epoch,
                                 extra={"val_dice": mean_dices})
                print(f"  saved new best checkpoint (mean dice {mean_overall:.4f})")

        save_checkpoint(Path(tcfg["checkpoint_dir"]) / "last.pt", model, optimizer, epoch)

    writer.close()


if __name__ == "__main__":
    main()
