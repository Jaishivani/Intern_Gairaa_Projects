import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn

from .inference import preprocess_image


def gradcam_overlay(
    image: Image.Image,
    model: nn.Module,
    class_index: int,
    alpha: float = 0.42,
) -> Image.Image:
    original = image.convert("RGB")
    device = next(model.parameters()).device
    image_tensor = preprocess_image(original).unsqueeze(0).to(device)
    captured: dict[str, torch.Tensor] = {}
    target_layer = model.layer4[-1]

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["activation"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__("gradient", gradient)
        )

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image_tensor.detach().requires_grad_(True))
            logits[0, class_index].backward()

        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = functional.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=(original.height, original.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap /= maximum

        intensity = heatmap.cpu().numpy()[..., None]
        red = np.clip(1.5 - np.abs(4.0 * intensity - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * intensity - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * intensity - 1.0), 0.0, 1.0)
        colors = np.concatenate((red, green, blue), axis=2)
        color_image = Image.fromarray((colors * 255).astype(np.uint8))
        return Image.blend(original, color_image, alpha)
    finally:
        hook.remove()
