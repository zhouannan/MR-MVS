import unittest

import torch

from models.alpha_fusion import AlphaFusion, AlphaFusionConfig


class AlphaFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 1
        self.hypotheses = 4
        self.height = 6
        self.width = 6
        self.intrinsics = torch.tensor(
            [[[100.0, 0.0, 2.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]]]
        )
        self.depth_values = torch.arange(1.0, 5.0)
        self.mono_depth = torch.full((1, self.height, self.width), 2.0)

    def test_probabilities_are_normalized(self) -> None:
        logits = torch.full(
            (self.batch, self.hypotheses, self.height, self.width),
            0.1,
        )
        fusion = AlphaFusion(
            AlphaFusionConfig(fixed_alpha=0.5, fixed_sigma_bins=1.0)
        )
        output = fusion(
            logits, self.mono_depth, self.depth_values, self.intrinsics
        )
        torch.testing.assert_close(
            output.probabilities.sum(dim=1),
            torch.ones(self.batch, self.height, self.width),
        )
        self.assertTrue(bool(output.prior_mask.all()))
        self.assertGreater(
            float((output.logits - logits).abs().sum()),
            0.0,
        )

    def test_invalid_mono_depth_leaves_logits_unchanged(self) -> None:
        logits = torch.randn(
            self.batch, self.hypotheses, self.height, self.width
        )
        invalid_mono = torch.full_like(self.mono_depth, float("inf"))
        fusion = AlphaFusion(
            AlphaFusionConfig(fixed_alpha=1.0, fixed_sigma_bins=1.0)
        )
        output = fusion(
            logits, invalid_mono, self.depth_values, self.intrinsics
        )
        torch.testing.assert_close(output.logits, logits)
        self.assertFalse(bool(output.prior_mask.any()))

    def test_depth_loss_reaches_alphanet(self) -> None:
        logits = torch.full(
            (
                self.batch,
                self.hypotheses,
                self.height,
                self.width,
            ),
            0.1,
            requires_grad=True,
        )
        fusion = AlphaFusion()
        output = fusion(
            logits, self.mono_depth, self.depth_values, self.intrinsics
        )
        loss = output.logits.square().mean()
        loss.backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in fusion.alpha_net.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_sum, 0.0)
        self.assertTrue(bool((output.sigma_bins > 0).all()))


if __name__ == "__main__":
    unittest.main()
