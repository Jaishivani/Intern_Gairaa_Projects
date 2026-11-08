from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from .config import (
    CLASS_FOLDERS,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    MODEL_PATH,
)



def build_model(pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def validation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return validation_transform()(image.convert("RGB"))


def load_model(
    checkpoint_path: Path = MODEL_PATH, device: torch.device | None = None
) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = device or torch.device("cpu")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("The checkpoint architecture is not ResNet18.")
    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError(f"The checkpoint image size must be {IMAGE_SIZE}.")

    saved_names = tuple(checkpoint.get("class_names", ()))
    if saved_names not in (CLASS_NAMES, CLASS_FOLDERS):
        raise ValueError("The checkpoint class order does not match NORMAL, POTHOLE.")
    saved_mapping = checkpoint.get("class_to_idx")
    if saved_mapping is not None and saved_mapping != CLASS_TO_INDEX:
        raise ValueError("The checkpoint class mapping does not match the application.")

    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def predict_image(
    image: Image.Image, model: nn.Module
) -> dict[str, Any]:
    device = next(model.parameters()).device
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = int(probabilities.argmax().item())

    return {
        "class_index": class_index,
        "class": CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index].item()),
        "probabilities": {
            CLASS_NAMES[index]: float(probabilities[index].item())
            for index in range(len(CLASS_NAMES))
        },
        "logits": logits[0].detach().cpu(),
    }
