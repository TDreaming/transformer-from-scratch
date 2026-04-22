"""Synthetic digit data generation used by the OCR demo."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


SEGMENTS_BY_DIGIT = {
    0: ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    1: ("upper_right", "lower_right"),
    2: ("top", "upper_right", "middle", "lower_left", "bottom"),
    3: ("top", "upper_right", "middle", "lower_right", "bottom"),
    4: ("upper_left", "upper_right", "middle", "lower_right"),
    5: ("top", "upper_left", "middle", "lower_right", "bottom"),
    6: ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    7: ("top", "upper_right", "lower_right"),
    8: ("top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom"),
    9: ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
}


@dataclass(frozen=True)
class SegmentBox:
    x0: int
    y0: int
    x1: int
    y1: int


def _draw_box(canvas: torch.Tensor, box: SegmentBox, value: float) -> None:
    """Draw a filled rectangle on the canvas."""
    height, width = canvas.shape
    x0 = max(0, min(width, box.x0))
    x1 = max(0, min(width, box.x1))
    y0 = max(0, min(height, box.y0))
    y1 = max(0, min(height, box.y1))
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] = torch.clamp_min(canvas[y0:y1, x0:x1], value)


def _segment_layout(thickness: int) -> dict[str, SegmentBox]:
    """Return seven-segment boxes for a 28x28 image."""
    left = 5
    right = 23
    top = 3
    middle = 13
    bottom = 23
    inner_left = 4
    inner_right = 24

    return {
        "top": SegmentBox(left, top, right, top + thickness),
        "upper_left": SegmentBox(inner_left, top + thickness, inner_left + thickness, middle),
        "upper_right": SegmentBox(inner_right - thickness, top + thickness, inner_right, middle),
        "middle": SegmentBox(left, middle - thickness // 2, right, middle + (thickness + 1) // 2),
        "lower_left": SegmentBox(inner_left, middle, inner_left + thickness, bottom),
        "lower_right": SegmentBox(inner_right - thickness, middle, inner_right, bottom),
        "bottom": SegmentBox(left, bottom, right, bottom + thickness),
    }


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _randfloat(generator: torch.Generator, low: float, high: float) -> float:
    value = torch.empty(1, dtype=torch.float32).uniform_(low, high, generator=generator)
    return float(value.item())


def render_digit(digit: int, generator: torch.Generator, image_size: int = 28) -> torch.Tensor:
    """Render a noisy synthetic digit image as a float32 tensor in [0, 1]."""
    if digit not in SEGMENTS_BY_DIGIT:
        raise ValueError(f"digit must be in 0-9, got {digit}")
    if image_size != 28:
        raise ValueError("This demo currently supports image_size=28 only.")

    canvas = torch.zeros((image_size, image_size), dtype=torch.float32)
    thickness = _randint(generator, 2, 5)
    shift_x = _randint(generator, -2, 3)
    shift_y = _randint(generator, -2, 3)
    intensity = _randfloat(generator, 0.75, 1.0)

    layout = _segment_layout(thickness)
    for name in SEGMENTS_BY_DIGIT[digit]:
        box = layout[name]
        jitter_x = _randint(generator, -1, 2)
        jitter_y = _randint(generator, -1, 2)
        shifted = SegmentBox(
            x0=box.x0 + shift_x + jitter_x,
            y0=box.y0 + shift_y + jitter_y,
            x1=box.x1 + shift_x + jitter_x,
            y1=box.y1 + shift_y + jitter_y,
        )
        _draw_box(canvas, shifted, intensity)

    # Add sparse noise so the model learns a slightly more realistic task.
    noise_mask = torch.rand(canvas.shape, generator=generator) < 0.03
    noise_values = torch.empty(canvas.shape, dtype=torch.float32).uniform_(0.1, 0.4, generator=generator)
    canvas = torch.where(noise_mask, torch.maximum(canvas, noise_values), canvas)
    canvas += torch.randn(canvas.shape, generator=generator) * 0.05
    canvas = canvas.clamp(0.0, 1.0)
    return canvas


class SyntheticDigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Balanced synthetic digit dataset generated on the fly."""

    def __init__(self, num_samples: int, seed: int = 0, image_size: int = 28) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if image_size != 28:
            raise ValueError("This demo currently supports image_size=28 only.")
        self.num_samples = num_samples
        self.seed = seed
        self.image_size = image_size
        labels = [index % 10 for index in range(num_samples)]
        shuffler = torch.Generator().manual_seed(seed)
        order = torch.randperm(num_samples, generator=shuffler).tolist()
        self.labels = [labels[index] for index in order]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        digit = self.labels[index]
        generator = torch.Generator().manual_seed(self.seed + index * 104_729)
        image_tensor = render_digit(digit, generator, image_size=self.image_size).unsqueeze(0)
        label_tensor = torch.tensor(digit, dtype=torch.long)
        return image_tensor, label_tensor
