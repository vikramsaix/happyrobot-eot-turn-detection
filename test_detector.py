from pathlib import Path
import unittest

from turn_detector.model import TurnDetector

ROOT = Path(__file__).resolve().parent


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = TurnDetector(ROOT / "artifacts" / "model.json")

    def test_short_pause_is_never_answered(self):
        out = self.detector.predict("That's all I needed, thanks.", 120, 300, "en")
        self.assertEqual(out["action"], "wait")
        self.assertIn("pause_below_minimum", out["reason_codes"])

    def test_unstable_transcript_is_never_answered(self):
        out = self.detector.predict("Yes, that works for me.", 650, 50, "en")
        self.assertEqual(out["action"], "wait")
        self.assertIn("transcript_not_stable", out["reason_codes"])

    def test_complete_utterance_can_be_answered(self):
        out = self.detector.predict("That's all I needed, thanks.", 850, 500, "en")
        self.assertEqual(out["action"], "respond")
        self.assertGreaterEqual(out["p_end"], out["threshold"])

    def test_open_clause_is_not_eagerly_answered(self):
        out = self.detector.predict("I need to reschedule because", 850, 500, "en")
        self.assertEqual(out["action"], "wait")

    def test_spanish_bonus_path(self):
        out = self.detector.predict("Eso es todo, muchas gracias.", 850, 500, "es")
        self.assertEqual(out["action"], "respond")

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.predict("Hello.", 500, 300, "fr")

    def test_response_has_monitoring_fields(self):
        out = self.detector.predict("Please call me tomorrow.", 500, 250, "en")
        self.assertTrue(0.0 <= out["p_end"] <= 1.0)
        self.assertTrue(out["model_version"])
        self.assertTrue(out["reason_codes"])


if __name__ == "__main__":
    unittest.main()
