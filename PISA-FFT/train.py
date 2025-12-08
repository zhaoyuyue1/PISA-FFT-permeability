from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from .data import build_dataloaders
from .models import FFTPermeabilityPredictorPatchPhysics
from .utils import seed_everything, count_parameters, medare


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    losses = []

    for images, patch_feats, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        patch_feats = patch_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images, patch_feats).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu())

    return float(torch.stack(losses).mean())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    losses = []
    labels, preds = [], []

    for images, patch_feats, targets in tqdm(loader, desc="Valid", leave=False):
        images = images.to(device)
        patch_feats = patch_feats.to(device)
        targets = targets.to(device)

        outputs = model(images, patch_feats).squeeze()
        loss = criterion(outputs, targets)

        losses.append(loss.detach().cpu())
        labels.append(targets.detach().cpu())
        preds.append(outputs.detach().cpu())

    val_loss = float(torch.stack(losses).mean())
    labels_np = torch.cat(labels, dim=0).numpy()
    preds_np = torch.cat(preds, dim=0).numpy()
    r2 = float(r2_score(labels_np, preds_np))
    return {"val_loss": val_loss, "val_r2": r2}


@torch.no_grad()
def test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    labels, preds = [], []

    for images, patch_feats, targets in tqdm(loader, desc="Test", leave=False):
        images = images.to(device)
        patch_feats = patch_feats.to(device)
        targets = targets.to(device)

        outputs = model(images, patch_feats).squeeze()
        labels.append(targets.detach().cpu())
        preds.append(outputs.detach().cpu())

    labels_np = torch.cat(labels, dim=0).numpy()
    preds_np = torch.cat(preds, dim=0).numpy()

    # Exponentiate back to the original permeability scale
    labels_exp = np.expm1(labels_np)
    preds_exp = np.expm1(preds_np)

    mse = float(mean_squared_error(labels_exp, preds_exp))
    r2 = float(r2_score(labels_exp, preds_exp))
    rmse = float(np.sqrt(mse))
    medare_val = medare(labels_exp, preds_exp)

    return {
        "Test_MSE": mse,
        "Test_R2": r2,
        "Test_RMSE": rmse,
        "Test_MedARE": medare_val,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FFT-based permeability prediction with physics-aware patches."
    )

    # Data paths
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing porous media images (PNG).")
    parser.add_argument("--csv-file", type=str, required=True,
                        help="CSV file with image IDs and permeability values.")

    # Data & model hyperparameters
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)

    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)

    # Misc
    parser.add_argument("--output-dir", type=str, default="./training_output/fft_external")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed. If not set, training is non-deterministic.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    if args.seed is not None:
        print(f"[Info] Using random seed = {args.seed}")
        seed_everything(args.seed)

    # Build dataloaders (60/20/20 split)
    train_loader, val_loader, test_loader = build_dataloaders(
        image_dir=args.image_dir,
        csv_file=args.csv_file,
        patch_size=args.patch_size,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train_ratio=0.6,
        val_ratio=0.2,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Model
    model = FFTPermeabilityPredictorPatchPhysics(
        image_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"[Info] Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_r2 = -1e9
    best_model_path = output_dir / "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch [{epoch}/{args.num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_r2"].append(val_metrics["val_r2"])

        print(
            f"  Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['val_loss']:.6f} | "
            f"Val R2: {val_metrics['val_r2']:.4f} | "
            f"Best Val R2: {best_r2:.4f}"
        )

        if val_metrics["val_r2"] > best_r2:
            best_r2 = val_metrics["val_r2"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model saved to {best_model_path}")
        else:
            print("  >> No improvement on validation R2.")

    # Save training log
    log_df = pd.DataFrame(history)
    log_df.to_csv(output_dir / "train_log.csv", index=False)

    # Test with the best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = test(model, test_loader, device)

    print("\n[Test Metrics]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
