import unittest

import torch

from models.stage_refine import (
    build_reproject_context,
    compute_multi_view_weighted_error_cached,
    refine_stage_depth_with_planes,
)


class GeometryRefinementTest(unittest.TestCase):
    def setUp(self):
        self.height = 8
        self.width = 8
        self.intrinsics = torch.tensor(
            [[[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]]]
        )
        self.projections = torch.zeros(1, 3, 2, 4, 4)
        self.projections[:, :, 0] = torch.eye(4)
        self.projections[:, :, 1] = torch.eye(4)
        self.projections[:, :, 1, :3, :3] = self.intrinsics[:, None]
        self.projections[:, 1:, 0, 0, 3] = -0.01

    def test_threshold_is_applied_per_source(self):
        reference = torch.full(
            (1, 1, self.height, self.width), 2.0
        )
        sources = [
            torch.full_like(reference, 2.0),
            torch.full_like(reference, 4.0),
        ]
        contexts = build_reproject_context(
            self.height,
            self.width,
            self.intrinsics,
            self.projections[:, 0, 0],
            [
                self.projections[:, 1, 1, :3, :3],
                self.projections[:, 2, 1, :3, :3],
            ],
            [
                self.projections[:, 1],
                self.projections[:, 2],
            ],
        )
        error, weight = compute_multi_view_weighted_error_cached(
            reference,
            sources,
            [
                torch.ones_like(reference),
                torch.ones_like(reference),
            ],
            contexts,
            threshold=0.01,
        )
        center = (0, 0, self.height // 2, self.width // 2)
        self.assertEqual(float(weight[center]), 1.0)
        self.assertLess(float(error[center]), 1e-4)

    def test_planar_depth_remains_stable(self):
        depths = torch.full((2, self.height, self.width), 2.0)
        refined = refine_stage_depth_with_planes(
            imgs=torch.zeros(1, 2, 3, self.height, self.width),
            proj_matrices={"stage1": self.projections[:, :2]},
            depths=depths,
            pre_output={
                "vis_list": [
                    torch.ones(1, 1, self.height, self.width)
                ]
            },
            depth_values=torch.tensor([[1.0, 3.0]]),
            stage_idx=0,
            reproj_threshold=0.01,
            refine_iters=2,
            use_noise=False,
            max_srcs=0,
        )
        torch.testing.assert_close(
            refined[:, 1:-1, 1:-1],
            depths[:1, 1:-1, 1:-1],
            atol=1e-4,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
