from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import BlastocystDataset, load_annotations, split_train_val
from .model import MultiHeadBlastocystNet, build_transforms


@dataclass
class EpochMetrics:
    loss: float
    expansion_acc: float
    icm_acc: float
    te_acc: float
    joint_acc: float


def _compute_batch_metrics(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, float]:
    pred_exp = outputs["expansion"].argmax(dim=1)
    pred_icm = outputs["icm"].argmax(dim=1)
    pred_te = outputs["te"].argmax(dim=1)

    exp_ok = (pred_exp == targets["expansion"]).float()
    icm_ok = (pred_icm == targets["icm"]).float()
    te_ok = (pred_te == targets["te"]).float()
    joint_ok = ((exp_ok == 1.0) & (icm_ok == 1.0) & (te_ok == 1.0)).float()

    return {
        "expansion_acc": float(exp_ok.mean().item()),
        "icm_acc": float(icm_ok.mean().item()),
        "te_acc": float(te_ok.mean().item()),
        "joint_acc": float(joint_ok.mean().item()),
    }


def _cross_entropy_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    return (
        criterion(outputs["expansion"], targets["expansion"])
        + criterion(outputs["icm"], targets["icm"])
        + criterion(outputs["te"], targets["te"])
    )


def _run_epoch(
    model: MultiHeadBlastocystNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_exp = 0.0
    total_icm = 0.0
    total_te = 0.0
    total_joint = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = {
            "expansion": batch["targets"]["expansion"].to(device),
            "icm": batch["targets"]["icm"].to(device),
            "te": batch["targets"]["te"].to(device),
        }

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = _cross_entropy_loss(outputs, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        stats = _compute_batch_metrics(outputs, targets)

        total_loss += float(loss.item())
        total_exp += stats["expansion_acc"]
        total_icm += stats["icm_acc"]
        total_te += stats["te_acc"]
        total_joint += stats["joint_acc"]
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("DataLoader produced zero batches.")

    return EpochMetrics(
        loss=total_loss / n_batches,
        expansion_acc=total_exp / n_batches,
        icm_acc=total_icm / n_batches,
        te_acc=total_te / n_batches,
        joint_acc=total_joint / n_batches,
    )


def train_model(
    annotations_csv: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    image_size: int = 224,
    seed: int = 42,
    pretrained: bool = True,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_csv)
    train_df, val_df = split_train_val(annotations, val_fraction=val_fraction, seed=seed)

    train_ds = BlastocystDataset(train_df, image_root=image_root, transform=build_transforms(image_size, train=True))
    val_ds = BlastocystDataset(val_df, image_root=image_root, transform=build_transforms(image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiHeadBlastocystNet(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_joint = -1.0
    best_checkpoint = output_path / "best_model.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_joint_acc": train_metrics.joint_acc,
                "val_loss": val_metrics.loss,
                "val_joint_acc": val_metrics.joint_acc,
                "val_expansion_acc": val_metrics.expansion_acc,
                "val_icm_acc": val_metrics.icm_acc,
                "val_te_acc": val_metrics.te_acc,
            }
        )

        if val_metrics.joint_acc > best_joint:
            best_joint = val_metrics.joint_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "meta": {
                    "epochs": epochs,
                    "best_epoch": epoch,
                    "best_val_joint_acc": best_joint,
                    "train_size": len(train_ds),
                    "val_size": len(val_ds),
                    "image_size": image_size,
                },
            }
            torch.save(checkpoint, best_checkpoint)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_joint_acc={val_metrics.joint_acc:.4f}"
        )

    (output_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_checkpoint
