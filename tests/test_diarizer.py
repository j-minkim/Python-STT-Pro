import unittest
import os
from unittest.mock import patch

from pyannote.audio import Pipeline

from diarizer import PyannoteDiarizer


class _FakePipeline:
    def to(self, _device):
        return None


class DiarizerCompatibilityTests(unittest.TestCase):
    def test_legacy_checkpoint_loading_is_enabled_for_pyannote(self):
        # Given: no process-wide PyTorch checkpoint override is configured.
        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)

        # When: the pyannote pipeline is loaded.
        def load_pipeline(*_args, **_kwargs):
            self.assertEqual(os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"), "1")
            return _FakePipeline()

        with patch.object(Pipeline, "from_pretrained", side_effect=load_pipeline):
            PyannoteDiarizer(hf_token="hf_test_token")

        # Then: the override does not leak to unrelated model loads.
        self.assertIsNone(os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"))


if __name__ == "__main__":
    unittest.main()
