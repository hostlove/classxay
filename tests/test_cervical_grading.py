"""Server-side regression checks for the paired input and ordinal grading contracts."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cervical.data import CervicalDataset, load_splits, prediction_cases
from cervical.engine import summarize
from cervical.inputs import prepare_case
from cervical.model import FeatureOrdinalHead, ordinal_logits_loss


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def add_pair(self, folder, stem):
        directory = self.root / folder
        directory.mkdir(exist_ok=True)
        y, x = np.indices((96, 80))
        image = ((2 * y + x) % 256).astype(np.uint8)
        foreground = (y >= 16) & (y < 80) & (x >= 20) & (x < 60)
        original = directory / f"{stem}.JPG"
        mask = directory / f"{stem}_seg.PNG"
        Image.fromarray(image).convert("RGB").save(original)
        # The RGB layer deliberately carries no radiograph information.
        segmented = Image.new("RGBA", (80, 96), color=(245, 245, 245, 0))
        segmented.putalpha(Image.fromarray(foreground.astype(np.uint8) * 255))
        segmented.save(mask)
        return original, mask, foreground

    def test_one_pair_one_grade_and_only_alpha_locates_the_original_pixels(self):
        original, _, foreground = self.add_pair("中度", "case01")
        cases = load_splits(self.root, ("train",), train_only=True)["train"]
        self.assertEqual(len(cases), 1)
        sample = CervicalDataset(cases)[0]
        self.assertEqual(sample["grade"], 2)
        with Image.open(original) as source:
            expected = prepare_case(np.asarray(source.convert("L")), foreground)
        for name in expected:
            torch.testing.assert_close(sample["inputs"][name], expected[name])
        self.assertEqual(sample["inputs"]["global_image"].shape, (3, 320, 320))
        self.assertEqual(sample["inputs"]["local_images"].shape, (6, 3, 192, 384))
        self.assertEqual(sample["inputs"]["source_intervals"].shape, (6, 2))
        self.assertEqual(sample["inputs"]["content_boxes"].shape, (6, 4))
        self.assertEqual(prediction_cases(self.root)[0].image, original)
        self.assertNotIn("grade", CervicalDataset(prediction_cases(original))[0])

    def test_bad_or_missing_mask_names_the_image(self):
        original, mask, _ = self.add_pair("1", "case01")
        Image.new("L", (8, 8)).save(mask)
        dataset = CervicalDataset(prediction_cases(original))
        with self.assertRaisesRegex(ValueError, "case01.*size"):
            dataset[0]
        mask.unlink()
        with self.assertRaisesRegex(ValueError, "missing.*case01"):
            prediction_cases(original)

    def test_manifest_validates_the_patient_across_all_splits(self):
        self.add_pair("0", "case01")
        self.add_pair("3", "case02")
        self.add_pair("2", "case03")
        manifest = self.root / "splits.csv"

        def write_manifest(last_patient):
            with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream).writerows([
                    ("image_path", "patient_id", "split"),
                    ("0/case01.JPG", "p1", "train"),
                    ("3/case02.JPG", "p2", "val"),
                    ("2/case03.JPG", last_patient, "test"),
                ])

        write_manifest("p1")
        with self.assertRaisesRegex(ValueError, "same patient"):
            load_splits(self.root, ("train", "val"), split_csv=manifest)
        write_manifest("p3")
        parts = load_splits(self.root, ("train", "val"), split_csv=manifest)
        self.assertEqual([c.grade for c in parts["train"]], [0])
        self.assertEqual([c.grade for c in parts["val"]], [3])


class OrdinalTests(unittest.TestCase):
    def test_ordered_probabilities_and_image_level_backward(self):
        head = FeatureOrdinalHead().eval()
        global_features = torch.randn(4, 384, requires_grad=True)
        aligned = torch.randn(4, 128, 128, requires_grad=True)
        valid = torch.ones(4, 128, dtype=torch.bool)
        result = head(global_features, aligned, valid, valid.long())
        self.assertEqual(result["logits"].shape, (4, 3))
        probabilities = result["probabilities"]
        self.assertEqual(probabilities.shape, (4, 4))
        self.assertTrue((result["thresholds"].diff() > 0).all())
        self.assertTrue((probabilities >= 0).all())
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(4))
        ordinal_logits_loss(result["logits"], torch.arange(4)).backward()
        for tensor in (global_features, aligned, head.threshold_gaps):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_grade_weights_scale_loss_even_for_one_image(self):
        logits = torch.tensor([[0.5, -0.5, -1.5]])
        grade_weights = torch.tensor([0.5, 1.0, 1.0, 2.0])
        for grade in (0, 3):
            labels = torch.tensor([grade], dtype=torch.uint8)
            unweighted = ordinal_logits_loss(logits, labels)
            weighted = ordinal_logits_loss(logits, labels, grade_weights=grade_weights)
            torch.testing.assert_close(weighted, unweighted * grade_weights[grade])

    def test_ordinal_metrics_and_undefined_kappa(self):
        metrics = summarize(np.eye(4, dtype=int), 0.1)
        self.assertEqual((metrics["accuracy"], metrics["macro_f1"], metrics["mae"], metrics["qwk"]),
                         (1.0, 1.0, 0.0, 1.0))
        counts = np.zeros((4, 4), dtype=int)
        counts[0, 3] = 2
        self.assertEqual(summarize(counts, 1.0)["mae"], 3.0)
        counts[0, 3], counts[0, 0] = 0, 2
        self.assertIsNone(summarize(counts, 0.1)["qwk"])


if __name__ == "__main__":
    unittest.main()
