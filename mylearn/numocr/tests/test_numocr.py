"""Unit tests for the NumOCR demo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


try:
    import torch

    from mylearn.numocr.data import SyntheticDigitDataset, render_digit
    from mylearn.numocr.model import SimpleDigitCNN
    from mylearn.numocr.predict import build_synthetic_sample, load_model, predict
    from mylearn.numocr.train import evaluate, train_one_epoch

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch is required to run numocr tests")
class NumOCRTests(unittest.TestCase):
    def test_render_digit_shape_and_range(self) -> None:
        image = render_digit(8, torch.Generator().manual_seed(123))
        self.assertEqual(tuple(image.shape), (28, 28))
        self.assertGreaterEqual(float(image.min().item()), 0.0)
        self.assertLessEqual(float(image.max().item()), 1.0)

    def test_dataset_returns_channel_first_tensor_and_label(self) -> None:
        dataset = SyntheticDigitDataset(num_samples=20, seed=7)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertEqual(label.dtype, torch.long)
        self.assertIn(int(label.item()), range(10))

    def test_dataset_validates_positive_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticDigitDataset(num_samples=0)

    def test_model_forward_output_shape(self) -> None:
        model = SimpleDigitCNN()
        batch = torch.rand(4, 1, 28, 28)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (4, 10))

    def test_single_epoch_training_and_evaluation_return_valid_metrics(self) -> None:
        dataset = SyntheticDigitDataset(num_samples=40, seed=1)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        model = SimpleDigitCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")

        train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, loader, criterion, device)

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(val_loss, 0.0)
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)

    def test_checkpoint_load_and_predict(self) -> None:
        model = SimpleDigitCNN()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

            loaded_model = load_model(checkpoint_path, torch.device("cpu"))
            sample = build_synthetic_sample(3, seed=5)
            predicted_digit, probabilities = predict(loaded_model, sample, torch.device("cpu"))

        self.assertIn(predicted_digit, range(10))
        self.assertEqual(tuple(probabilities.shape), (10,))
        self.assertAlmostEqual(float(probabilities.sum().item()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
