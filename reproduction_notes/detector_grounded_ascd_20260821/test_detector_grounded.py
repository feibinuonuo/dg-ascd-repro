#!/usr/bin/env python3
import unittest

import torch

from ascd_detector_grounded import (
    apply_detector_object_mask,
    canonical_chair_object,
    terminal_decoded_chair_object,
)


class FakeTokenizer:
    def __init__(self, text_by_id):
        self.text_by_id = text_by_id

    def decode(self, ids, **_kwargs):
        return self.text_by_id[int(ids[-1])]

    def convert_ids_to_tokens(self, token_id):
        return f"tok_{int(token_id)}"


class DetectorGroundedTests(unittest.TestCase):
    def test_chair_aliases(self):
        self.assertEqual(canonical_chair_object("bike"), "bicycle")
        self.assertEqual(canonical_chair_object("woman"), "person")
        self.assertIsNone(canonical_chair_object("running"))

    def test_unsupported_object_is_masked_to_next_ascd_token(self):
        tokenizer = FakeTokenizer({0: " a cat", 1: " runs", 2: " a dog"})
        scores = torch.tensor([[4.0, 3.0, 2.0]])
        masked, event = apply_detector_object_mask(
            scores, tokenizer=tokenizer, generated_token_ids=[],
            support_scores={"cat": 0.1, "dog": 0.9}, threshold=0.2, top_k=2,
        )
        self.assertEqual(int(torch.argmax(masked, dim=-1)[0]), 1)
        self.assertEqual(event["masked_token_ids"], [0])
        self.assertTrue(event["selection_changed"])

    def test_supported_object_and_non_object_are_unchanged(self):
        tokenizer = FakeTokenizer({0: " a cat", 1: " runs"})
        scores = torch.tensor([[4.0, 3.0]])
        masked, event = apply_detector_object_mask(
            scores, tokenizer=tokenizer, generated_token_ids=[],
            support_scores={"cat": 0.3}, threshold=0.2, top_k=2,
        )
        self.assertTrue(torch.equal(masked, scores))
        self.assertEqual(event["masked_token_ids"], [])
        self.assertFalse(event["selection_changed"])

    def test_punctuation_and_special_tokens_are_not_object_candidates(self):
        tokenizer = FakeTokenizer({0: ".", 1: "</s>", 2: " a cat"})
        self.assertIsNone(terminal_decoded_chair_object(tokenizer, [2], 0))
        self.assertIsNone(terminal_decoded_chair_object(tokenizer, [2], 1))
        scores = torch.tensor([[4.0, 3.0]])
        masked, event = apply_detector_object_mask(
            scores, tokenizer=tokenizer, generated_token_ids=[2],
            support_scores={"cat": 0.1}, threshold=0.2, top_k=2,
        )
        self.assertTrue(torch.equal(masked, scores))
        self.assertEqual(event["object_candidates"], [])

    def test_all_finite_protection_preserves_distribution(self):
        tokenizer = FakeTokenizer({0: " a cat", 1: " a dog"})
        scores = torch.tensor([[4.0, 3.0]])
        masked, event = apply_detector_object_mask(
            scores, tokenizer=tokenizer, generated_token_ids=[],
            support_scores={"cat": 0.1, "dog": 0.1}, threshold=0.2, top_k=2,
        )
        self.assertTrue(torch.equal(masked, scores))
        self.assertTrue(event["protected_no_finite"])
        self.assertEqual(event["masked_token_ids"], [])


if __name__ == "__main__":
    unittest.main()
