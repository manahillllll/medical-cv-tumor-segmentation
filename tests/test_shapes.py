"""
Sanity tests that run without any BraTS data — pure shape/logic checks on synthetic
tensors. These are the milestone-2-style "does the wiring work" checks: they don't
tell you whether the model is any good, only that the forward/backward passes,
stitching, and uncertainty/explainability math are implemented correctly.

Run with: pytest tests/test_shapes.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gradcam3d import GradCAM3D
from src.inference import sliding_window_inference
from src.losses import DiceCELoss, SoftDiceLoss, compute_class_weights, one_hot
from src.metrics import brats_region_dice, dice_score_per_class, expected_calibration_error
from src.models.classifier import SegClassifier
from src.models.unet3d import UNet3D
from src.uncertainty import ensemble_predict, mc_dropout_predict


def make_unet(**overrides):
    defaults = dict(in_channels=4, num_classes=4, base_filters=4, depth=2, dropout_p=0.2)
    defaults.update(overrides)
    return UNet3D(**defaults)


def test_unet3d_forward_shape():
    model = make_unet()
    x = torch.randn(2, 4, 32, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 4, 32, 32, 32)


def test_unet3d_returns_bottleneck_features():
    model = make_unet()
    x = torch.randn(1, 4, 32, 32, 32)
    logits, feats = model(x, return_features=True)
    assert feats.shape[1] == model.bottleneck_channels
    assert feats.shape[0] == 1


def test_unet3d_nondivisible_input_shape():
    """Patch sizes not evenly divisible by 2**depth should still round-trip correctly."""
    model = make_unet(depth=3)
    x = torch.randn(1, 4, 33, 33, 33)
    logits = model(x)
    assert logits.shape == (1, 4, 33, 33, 33)


def test_dice_ce_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    model = make_unet(num_classes=3)
    x = torch.randn(1, 4, 16, 16, 16)
    y = torch.randint(0, 3, (1, 16, 16, 16))
    criterion = DiceCELoss(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_soft_dice_loss_perfect_prediction_near_zero():
    num_classes = 3
    targets = torch.randint(0, num_classes, (1, 8, 8, 8))
    logits = one_hot(targets, num_classes) * 20.0 - 10.0  # confident correct logits
    loss = SoftDiceLoss(num_classes=num_classes)(logits, targets)
    assert loss.item() < 0.05


def test_compute_class_weights_shape_and_positivity():
    labels = torch.randint(0, 4, (2, 16, 16, 16))
    weights = compute_class_weights(labels, num_classes=4)
    assert weights.shape == (4,)
    assert (weights > 0).all()


def test_sliding_window_inference_reconstructs_full_shape():
    model = make_unet(base_filters=4, depth=2).eval()
    volume = torch.randn(4, 40, 48, 56)
    probs = sliding_window_inference(volume, model, patch_size=(24, 24, 24),
                                      num_classes=4, device="cpu", overlap=0.5)
    assert probs.shape == (4, 40, 48, 56)
    # softmax outputs should sum to ~1 across the class dimension everywhere
    sums = probs.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_sliding_window_inference_smaller_than_patch():
    model = make_unet(base_filters=4, depth=2).eval()
    volume = torch.randn(4, 10, 10, 10)
    probs = sliding_window_inference(volume, model, patch_size=(24, 24, 24),
                                      num_classes=4, device="cpu")
    assert probs.shape == (4, 10, 10, 10)


def test_mc_dropout_predict_shapes():
    model = make_unet(base_filters=4, depth=2, dropout_p=0.5)
    x = torch.randn(1, 4, 16, 16, 16)
    out = mc_dropout_predict(model, x, n_samples=5)
    assert out["mean_probs"].shape == (1, 4, 16, 16, 16)
    assert out["epistemic_uncertainty"].shape == (1, 16, 16, 16)
    assert (out["epistemic_uncertainty"] >= 0).all()


def test_mc_dropout_variance_nonzero_with_dropout_active():
    model = make_unet(base_filters=4, depth=2, dropout_p=0.5)
    x = torch.randn(1, 4, 16, 16, 16)
    out = mc_dropout_predict(model, x, n_samples=10)
    assert out["epistemic_uncertainty"].sum() > 0, "MC dropout produced zero variance across samples"


def test_ensemble_predict_shapes():
    models = [make_unet(base_filters=4, depth=2) for _ in range(3)]
    x = torch.randn(1, 4, 16, 16, 16)
    out = ensemble_predict(models, x)
    assert out["mean_probs"].shape == (1, 4, 16, 16, 16)
    assert out["all_samples"].shape[0] == 3


def test_seg_classifier_forward_shapes():
    unet = make_unet(num_classes=4)
    model = SegClassifier(unet, num_classification_classes=2)
    x = torch.randn(1, 4, 32, 32, 32)
    seg_logits, cls_logits = model(x)
    assert seg_logits.shape == (1, 4, 32, 32, 32)
    assert cls_logits.shape == (1, 2)


def test_seg_classifier_frozen_encoder_blocks_gradients():
    unet = make_unet(num_classes=4)
    model = SegClassifier(unet, num_classification_classes=2, freeze_encoder=True)
    x = torch.randn(1, 4, 32, 32, 32)
    _, cls_logits = model(x)
    loss = cls_logits.sum()
    loss.backward()
    unet_grad_norms = [p.grad.abs().sum().item() for p in unet.parameters() if p.grad is not None]
    assert sum(unet_grad_norms) == 0, "encoder received gradients despite freeze_encoder=True"
    head_grad_norms = [p.grad.abs().sum().item() for p in model.classifier.parameters() if p.grad is not None]
    assert sum(head_grad_norms) > 0, "classification head received no gradient"


def test_gradcam3d_output_shape_and_range():
    unet = make_unet(num_classes=4, depth=2)
    model = SegClassifier(unet, num_classification_classes=2)
    target_layer = model.unet.encoders[-1].conv.block[3]
    cam = GradCAM3D(model, target_layer)
    x = torch.randn(1, 4, 32, 32, 32)
    heatmap = cam(x, class_idx=0)
    assert heatmap.shape == (32, 32, 32)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


def test_dice_score_per_class_perfect_match():
    pred = torch.randint(0, 4, (10, 10, 10))
    scores = dice_score_per_class(pred, pred.clone(), num_classes=4)
    assert torch.allclose(scores, torch.ones(4), atol=1e-3)


def test_brats_region_dice_perfect_match():
    label = torch.randint(0, 4, (10, 10, 10))
    regions = brats_region_dice(label, label.clone())
    for v in regions.values():
        assert v == pytest.approx(1.0, abs=1e-3)


def test_expected_calibration_error_perfect_calibration_is_zero():
    import numpy as np
    confidences = np.array([0.9] * 100)
    correct = np.array([True] * 90 + [False] * 10)
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)
