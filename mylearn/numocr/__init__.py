"""A tiny PyTorch OCR example for digit classification."""

from .data import SyntheticDigitDataset, render_digit
from .model import SimpleDigitCNN

__all__ = ["SimpleDigitCNN", "SyntheticDigitDataset", "render_digit"]
