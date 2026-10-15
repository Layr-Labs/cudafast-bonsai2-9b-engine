#!/usr/bin/env python3
"""Unit tests for oracle stop_type handling (no network, no GPU)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# oracle_serial_check lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_serial_check as osc  # noqa: E402

N = 128
PROMPT = [1, 2, 3]


def _completion(tokens: list[int], stop_type: str | None) -> dict:
    body: dict = {"tokens": tokens}
    if stop_type is not None:
        body["stop_type"] = stop_type
    return body


class ServerGreedyStopType(unittest.TestCase):
    @patch.object(osc, "http_json")
    def test_returns_stop_type_from_completion(self, mock_http_json):
        mock_http_json.return_value = _completion([10, 11], "eos")
        tokens, stop_type = osc.server_greedy("http://x", PROMPT, N)
        self.assertEqual(tokens, [10, 11])
        self.assertEqual(stop_type, "eos")
        mock_http_json.assert_called_once()
        url = mock_http_json.call_args[0][0]
        self.assertTrue(url.endswith("/completion"))


class ResidentForCompare(unittest.TestCase):
    def test_eos_short_oracle_cuts_resident(self):
        oracle = [1] * 50
        resident = [1] * 50 + [9] * 78
        trimmed, err = osc.resident_for_compare(oracle, resident, "eos", N)
        self.assertIsNone(err)
        self.assertEqual(trimmed, resident[:50])

    def test_limit_full_oracle_no_cut(self):
        oracle = [1] * N
        resident = [1] * N
        trimmed, err = osc.resident_for_compare(oracle, resident, "limit", N)
        self.assertIsNone(err)
        self.assertEqual(trimmed, resident)

    def test_limit_short_oracle_fails(self):
        oracle = [1] * 40
        resident = [1] * N
        trimmed, err = osc.resident_for_compare(oracle, resident, "limit", N)
        self.assertIsNotNone(err)
        self.assertIn("limit", err)
        self.assertEqual(trimmed, resident)

    def test_empty_oracle_eos_fails(self):
        oracle: list[int] = []
        resident = [7] * N
        trimmed, err = osc.resident_for_compare(oracle, resident, "eos", N)
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())
        self.assertEqual(trimmed, resident)

    def test_word_short_oracle_fails(self):
        oracle = [1] * 40
        resident = [1] * 40 + [0] * 88
        trimmed, err = osc.resident_for_compare(oracle, resident, "word", N)
        self.assertIsNotNone(err)
        self.assertIn("word", err)
        self.assertEqual(trimmed, resident)

    def test_none_short_oracle_fails(self):
        oracle = [1] * 40
        resident = [1] * 40
        trimmed, err = osc.resident_for_compare(oracle, resident, "none", N)
        self.assertIsNotNone(err)
        self.assertIn("none", err)

    def test_missing_stop_type_short_oracle_fails(self):
        oracle = [1] * 40
        resident = [1] * 40
        trimmed, err = osc.resident_for_compare(oracle, resident, None, N)
        self.assertIsNotNone(err)
        self.assertIn("stop_type", err)


class MainEarlyStopFailure(unittest.TestCase):
    """main() must exit 1 when resident_for_compare rejects a prompt (early_stop path)."""

    @patch.object(osc, "resident_greedy")
    @patch.object(osc, "server_greedy")
    @patch.object(osc, "fit_tokens")
    @patch.object(osc, "load_prompt_sources")
    @patch.object(osc, "start_server")
    @patch.object(osc.os.path, "isfile", return_value=True)
    @patch.object(osc.os, "access", return_value=True)
    def test_main_nonzero_on_early_stop(
        self,
        _mock_access,
        _mock_isfile,
        mock_start_server,
        mock_load_prompt_sources,
        mock_fit_tokens,
        mock_server_greedy,
        mock_resident_greedy,
    ):
        mock_start_server.return_value = ("http://127.0.0.1:1", None, None)
        mock_load_prompt_sources.return_value = [(f"p{i}", "x") for i in range(6)]
        mock_fit_tokens.return_value = PROMPT
        mock_server_greedy.return_value = ([1, 2], "word")
        mock_resident_greedy.return_value = [1, 2] + [0] * (N - 2)

        rc = osc.main()

        self.assertEqual(rc, 1)
        mock_start_server.assert_called_once()
        mock_load_prompt_sources.assert_called_once()
        self.assertEqual(mock_fit_tokens.call_count, 6)
        self.assertEqual(mock_server_greedy.call_count, 6)
        self.assertEqual(mock_resident_greedy.call_count, 6)


if __name__ == "__main__":
    unittest.main()
