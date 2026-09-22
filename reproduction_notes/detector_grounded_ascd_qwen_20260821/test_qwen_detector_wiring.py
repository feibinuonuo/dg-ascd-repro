#!/usr/bin/env python3
"""Regression checks for default-off Qwen detector wiring."""

import inspect
import unittest

from ascd_utils_v3_qwen.contrastive_sample import _sample


class QwenDetectorWiringTests(unittest.TestCase):
    def test_sampling_mode_is_bound_before_detector_guard(self):
        source = inspect.getsource(_sample)
        self.assertLess(
            source.index("do_sample = generation_config.do_sample"),
            source.index("detector_grounded = bool("),
        )

    def test_detector_masks_only_postprocessed_scores_before_selection(self):
        source = inspect.getsource(_sample)
        rerank = source.index("next_token_scores = _context_entropy_rerank(")
        detector = source.index("if detector_grounded:", rerank)
        selection = source.index("# token selection", detector)
        self.assertLess(rerank, detector)
        self.assertLess(detector, selection)
        self.assertIn("apply_detector_object_mask(", source[detector:selection])


if __name__ == "__main__":
    unittest.main()
