"""Run inference with the trained digit OCR model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mylearn.numocr.data import render_digit
from mylearn.numocr.model import SimpleDigitCNN
from mylearn.numocr.train import DEFAULT_CHECKPOINT, choose_device

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency for local image prediction
    Image = None


def load_model(checkpoint_path: Path, device: torch.device) -> SimpleDigitCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleDigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required for --image prediction. Install it with `pip install pillow`.")

    image = Image.open(image_path).convert("L").resize((28, 28))
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).view(28, 28) / 255.0
    if pixels.mean().item() > 0.5:
        pixels = 1.0 - pixels
    tensor = pixels.unsqueeze(0).unsqueeze(0)
    return tensor


def build_synthetic_sample(digit: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return render_digit(digit, generator).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict(model: SimpleDigitCNN, image_tensor: torch.Tensor, device: torch.device) -> tuple[int, torch.Tensor]:
    logits = model(image_tensor.to(device))
    probabilities = torch.softmax(logits, dim=1).cpu()[0]
    predicted_digit = int(torch.argmax(probabilities).item())
    return predicted_digit, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a digit using the trained OCR model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to a local digit image.")
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        help="Generate a synthetic sample for this digit and predict it.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed used for synthetic sample generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.digit is None:
        raise SystemExit("Please provide --image or --digit.")
    if args.digit is not None and not 0 <= args.digit <= 9:
        raise SystemExit("--digit must be between 0 and 9.")
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run `python3 -m mylearn.numocr.train` first."
        )

    device = choose_device()
    model = load_model(args.checkpoint, device)

    if args.image is not None:
        image_tensor = preprocess_image(args.image)
        source = f"image={args.image}"
    else:
        image_tensor = build_synthetic_sample(args.digit, args.seed)
        source = f"synthetic digit={args.digit}, seed={args.seed}"

    predicted_digit, probabilities = predict(model, image_tensor, device)
    top3 = torch.topk(probabilities, k=3)

    print(f"Source: {source}")
    print(f"Predicted digit: {predicted_digit}")
    print("Top-3 probabilities:")
    for score, label in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
