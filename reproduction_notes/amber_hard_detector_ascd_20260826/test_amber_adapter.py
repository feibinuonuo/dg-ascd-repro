#!/usr/bin/env python3
"""CPU-only unit tests for the AMBER adapter's data and default-off invariants."""

from __future__ import annotations

import unittest

from experiments_v3.eval.model_vqa_amber_detector import get_chunk, validate_queries


class AmberAdapterTests(unittest.TestCase):
    def test_valid_official_rows(self):
        rows = [
            {"id": 1, "image": "AMBER_1.jpg", "query": "Describe this image."},
            {"id": 2, "image": "AMBER_2.jpg", "query": "Describe this image."},
        ]
        self.assertEqual(list(validate_queries(rows)), [1, 2])

    def test_bad_naming_and_duplicate_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_queries([{ "id": 1, "image": "AMBER_2.jpg", "query": "x" }])
        with self.assertRaises(ValueError):
            validate_queries([
                {"id": 1, "image": "AMBER_1.jpg", "query": "x"},
                {"id": 1, "image": "AMBER_1.jpg", "query": "x"},
            ])

    def test_chunk_bounds(self):
        self.assertEqual(get_chunk([1, 2, 3], 2, 0), [1, 2])
        with self.assertRaises(ValueError):
            get_chunk([1], 1, 1)


if __name__ == "__main__":
    unittest.main()
