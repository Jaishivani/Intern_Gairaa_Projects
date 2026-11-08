import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    DATASET_PATH,
    IMAGE_SIZE,
    MODEL_PATH,
    get_device,
)
from src.inference import build_model, training_transform, validation_transform


ROUTE_PATTERN = re.compile(
    r"_(?P<state>[A-Z]{2})_(?P<road>\d+)_(?P<route>\d+[A-Z]{2,3}\d+)_"
)
FALSE_POSITIVE_PATH = PROJECT_ROOT / "outputs" / "false_positives"
FALSE_NEGATIVE_PATH = PROJECT_ROOT / "outputs" / "false_negatives"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Roadwatch ResNet18 classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def route_group(image_path: str | Path) -> str:
    source_name = Path(image_path).name.split("__", 1)[-1]
    match = ROUTE_PATTERN.search(source_name)
    if match is None:
        raise ValueError(f"Cannot parse source route from {source_name}.")
    return "/".join(match.group(key) for key in ("state", "road", "route"))


def audit_dataset(datasets: dict[str, ImageFolder]) -> None:
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    route_splits: dict[str, set[str]] = defaultdict(set)
    formats: Counter[str] = Counter()
    corrupt_images: list[str] = []

    for split_name, dataset in datasets.items():
        if not dataset.samples:
            raise SystemExit(f"Dataset split {split_name!r} contains no images.")
        if dataset.class_to_idx != CLASS_TO_INDEX:
            raise SystemExit(
                f"{split_name} must map normal=0 and pothole=1; "
                f"found {dataset.class_to_idx}."
            )

        for image_path, target in dataset.samples:
            label = CLASS_FOLDERS[target]
            route_splits[route_group(image_path)].add(split_name)
            digest = hashlib.sha256()
            try:
                with Path(image_path).open("rb") as image_file:
                    for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                with Image.open(image_path) as image:
                    image.verify()
                    formats[image.format or "unknown"] += 1
            except (OSError, ValueError, Image.DecompressionBombError):
                corrupt_images.append(image_path)
                continue
            hashes[digest.hexdigest()].append((split_name, label, image_path))

    if corrupt_images:
        examples = ", ".join(corrupt_images[:5])
        raise SystemExit(
            f"Found {len(corrupt_images)} unreadable image(s); repair/remove them first: {examples}"
        )

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    if duplicate_groups:
        examples = "; ".join(
            " = ".join(item[2] for item in items[:3])
            for items in duplicate_groups[:3]
        )
        raise SystemExit(
            f"Found {len(duplicate_groups)} exact duplicate image group(s): {examples}"
        )

    overlapping_routes = {
        route: sorted(splits)
        for route, splits in route_splits.items()
        if len(splits) > 1
    }
    if overlapping_routes:
        raise SystemExit(
            "Road-route leakage detected across splits: "
            f"{list(overlapping_routes.items())[:5]}"
        )

    class_counts = {
        split_name: Counter(dataset.targets)
        for split_name, dataset in datasets.items()
    }
    for split_name, counts in class_counts.items():
        if any(counts[index] == 0 for index in range(len(CLASS_NAMES))):
            raise SystemExit(f"{split_name} must contain both normal and pothole images.")
        print(f"{split_name.title()} images:")
        print(f"Normal: {counts[CLASS_TO_INDEX['normal']]}")
        print(f"Pothole: {counts[CLASS_TO_INDEX['pothole']]}")

    print(f"Image formats: {dict(formats)}")
    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Route groups: {len(route_splits)}; split overlap: 0")


def classification_metrics(actual: torch.Tensor, predicted: torch.Tensor) -> dict[str, Any]:
    encoded = actual * len(CLASS_NAMES) + predicted
    confusion = torch.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(
        len(CLASS_NAMES), len(CLASS_NAMES)
    )
    true_positive = confusion.diag().float()
    precision_per_class = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall_per_class = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class).clamp_min(1e-12)
    )
    pothole_index = CLASS_TO_INDEX["pothole"]
    return {
        "accuracy": float(true_positive.sum().item() / confusion.sum().item()),
        "balanced_accuracy": float(recall_per_class.mean().item()),
        "precision": float(precision_per_class.mean().item()),
        "recall": float(recall_per_class.mean().item()),
        "f1": float(f1_per_class.mean().item()),
        "pothole_precision": float(precision_per_class[pothole_index].item()),
        "pothole_recall": float(recall_per_class[pothole_index].item()),
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    if training:
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

    total_loss = 0.0
    total = 0
    actual_batches = []
    predicted_batches = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        actual_batches.append(labels.detach().cpu())
        predicted_batches.append(logits.detach().argmax(dim=1).cpu())

    metrics = classification_metrics(
        torch.cat(actual_batches), torch.cat(predicted_batches)
    )
    return total_loss / total, metrics


def evaluate_dataset(
    model: nn.Module,
    dataset: ImageFolder,
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    actual_batches = []
    predicted_batches = []
    confidence_batches = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels.to(device))
            predictions = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            actual_batches.append(labels)
            predicted_batches.append(predictions.cpu())
            confidence_batches.append(confidence.cpu())

    actual = torch.cat(actual_batches)
    predicted = torch.cat(predicted_batches)
    confidences = torch.cat(confidence_batches)
    metrics = classification_metrics(actual, predicted)
    rows = []
    for index, ((image_path, target), predicted_index, confidence) in enumerate(
        zip(dataset.samples, predicted.tolist(), confidences.tolist())
    ):
        rows.append(
            {
                "filename": Path(image_path).name,
                "actual": CLASS_NAMES[target],
                "predicted": CLASS_NAMES[predicted_index],
                "confidence": confidence,
                "correct": target == predicted_index,
                "actual_index": target,
                "predicted_index": predicted_index,
            }
        )
    return total_loss / total, metrics, rows


def print_metrics(split_name: str, loss: float, metrics: dict[str, Any]) -> None:
    print(
        f"{split_name} loss: {loss:.4f} | accuracy: {metrics['accuracy']:.2%} | "
        f"precision: {metrics['precision']:.2%} | recall: {metrics['recall']:.2%} | "
        f"F1: {metrics['f1']:.2%} | balanced accuracy: {metrics['balanced_accuracy']:.2%}"
    )
    print(
        f"Pothole precision: {metrics['pothole_precision']:.2%} | "
        f"Pothole recall: {metrics['pothole_recall']:.2%}"
    )
    matrix = metrics["confusion_matrix"]
    print("Confusion matrix (actual rows, predicted columns):")
    print(f"                 {CLASS_NAMES[0]:<14} {CLASS_NAMES[1]}")
    for class_name, row in zip(CLASS_NAMES, matrix):
        print(f"Actual {class_name:<14} {row[0]:<14} {row[1]}")
    print(
        f"False positives: {matrix[CLASS_TO_INDEX['normal']][CLASS_TO_INDEX['pothole']]} | "
        f"False negatives: {matrix[CLASS_TO_INDEX['pothole']][CLASS_TO_INDEX['normal']]}"
    )


def save_prediction_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("filename", "actual", "predicted", "confidence", "correct")
    with path.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def save_test_errors(
    dataset: ImageFolder,
    rows: list[dict[str, Any]],
) -> None:
    for directory in (FALSE_POSITIVE_PATH, FALSE_NEGATIVE_PATH):
        directory.mkdir(parents=True, exist_ok=True)
        for old_image in directory.glob("*.jpg"):
            old_image.unlink()

    for (image_path, _), row in zip(dataset.samples, rows):
        if row["actual_index"] == CLASS_TO_INDEX["normal"] and row["predicted_index"] == CLASS_TO_INDEX["pothole"]:
            shutil.copy2(image_path, FALSE_POSITIVE_PATH / Path(image_path).name)
        elif row["actual_index"] == CLASS_TO_INDEX["pothole"] and row["predicted_index"] == CLASS_TO_INDEX["normal"]:
            shutil.copy2(image_path, FALSE_NEGATIVE_PATH / Path(image_path).name)


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"
    for split_path in (train_path, val_path, test_path):
        missing = [name for name in CLASS_FOLDERS if not (split_path / name).is_dir()]
        if missing:
            raise SystemExit(
                "Dataset is required before training. Add real images under "
                f"{split_path / CLASS_FOLDERS[0]} and {split_path / CLASS_FOLDERS[1]}."
            )

    train_data = ImageFolder(train_path, transform=training_transform())
    val_data = ImageFolder(val_path, transform=validation_transform())
    test_data = ImageFolder(test_path, transform=validation_transform())
    audit_dataset({"training": train_data, "validation": val_data, "test": test_data})

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    print(f"Device: {device}")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = build_model(pretrained=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer3.parameters():
        parameter.requires_grad = True
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    class_counts = torch.bincount(
        torch.tensor(train_data.targets), minlength=len(CLASS_NAMES)
    ).float()
    class_weights = (
        len(train_data.targets) / (len(CLASS_NAMES) * class_counts)
    ).pow(0.75).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_balanced_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"\nEpoch {epoch + 1}/{args.epochs} | learning rate: {optimizer.param_groups[0]['lr']:.2g}")
        print_metrics("Training", train_loss, train_metrics)
        print_metrics("Validation", val_loss, val_metrics)

        if val_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics["balanced_accuracy"]
            epochs_without_improvement = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "resnet18",
                    "class_names": CLASS_NAMES,
                    "class_to_idx": CLASS_TO_INDEX,
                    "image_size": IMAGE_SIZE,
                    "validation_metrics": val_metrics,
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                },
                output_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    if not output_path.is_file():
        raise SystemExit(f"Training did not create a model checkpoint at {output_path}.")

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    for split_name, dataset in (("Validation", val_data), ("Test", test_data)):
        loss, metrics, rows = evaluate_dataset(model, dataset, device, args.batch_size)
        print(f"\nFinal {split_name} results:")
        print_metrics(split_name, loss, metrics)
        report_name = "validation_predictions.csv" if split_name == "Validation" else "test_predictions.csv"
        save_prediction_report(PROJECT_ROOT / "outputs" / report_name, rows)
        if split_name == "Test":
            save_test_errors(dataset, rows)

    print("TRAINING COMPLETE")
    print("Model saved to:")
    print(output_path)
    print("Evaluation reports saved to:")
    print(PROJECT_ROOT / "outputs")


if __name__ == "__main__":
    main()
