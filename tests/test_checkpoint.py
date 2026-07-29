import unittest

import torch

from models.alpha_fusion import AlphaNet
from models.checkpoint import alpha_keys, checkpoint_state


class CheckpointTest(unittest.TestCase):
    def test_extract_alpha_weights_and_strip_module_prefix(self) -> None:
        source = AlphaNet()
        synthetic_state = {}
        for stage_index in range(4):
            for key, value in source.state_dict().items():
                synthetic_state[
                    f"module.fusions.{stage_index}.alpha_net.{key}"
                ] = value.clone() + stage_index

        normalized = checkpoint_state({"state_dict": synthetic_state})
        extracted = alpha_keys(normalized, stages=(0, 1, 2))
        self.assertEqual(len(extracted), 24)
        for key, value in source.state_dict().items():
            torch.testing.assert_close(
                extracted[f"fusions.2.alpha_net.{key}"],
                value + 2,
            )


if __name__ == "__main__":
    unittest.main()
