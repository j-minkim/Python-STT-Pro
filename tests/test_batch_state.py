import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_state import BatchState, source_key_for_path, source_key_for_url
from media_scan import collect_supported_files


class BatchStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir = os.path.join(self.tmp.name, 'state')

    def tearDown(self):
        self.tmp.cleanup()

    def make_state(self, key='dir:/some/folder'):
        return BatchState(key, state_dir=self.state_dir)

    def test_done_persists_across_instances(self):
        state = self.make_state()
        self.assertFalse(state.is_done('a.mp3|10|1'))
        state.mark_done('a.mp3|10|1', outputs={'txt': '/out/a.txt'})

        reloaded = self.make_state()
        self.assertTrue(reloaded.is_done('a.mp3|10|1'))
        self.assertFalse(reloaded.is_done('b.mp3|10|1'))

    def test_failed_files_are_retried(self):
        state = self.make_state()
        state.mark_failed('a.mp3|10|1', 'boom')
        reloaded = self.make_state()
        self.assertFalse(reloaded.is_done('a.mp3|10|1'))
        self.assertEqual(reloaded.data['files']['a.mp3|10|1']['status'], 'failed')

    def test_reset_clears_progress(self):
        state = self.make_state()
        state.mark_done('a.mp3|10|1')
        state.reset()
        reloaded = self.make_state()
        self.assertFalse(reloaded.is_done('a.mp3|10|1'))

    def test_corrupt_state_file_starts_fresh(self):
        state = self.make_state()
        state.mark_done('a.mp3|10|1')
        with open(state.path, 'w', encoding='utf-8') as f:
            f.write('{not valid json')
        reloaded = self.make_state()
        self.assertFalse(reloaded.is_done('a.mp3|10|1'))
        reloaded.mark_done('b.mp3|5|2')
        self.assertTrue(self.make_state().is_done('b.mp3|5|2'))

    def test_different_sources_do_not_collide(self):
        first = self.make_state('dir:/folder/one')
        first.mark_done('a.mp3|10|1')
        second = self.make_state('dir:/folder/two')
        self.assertFalse(second.is_done('a.mp3|10|1'))

    def test_file_key_changes_when_file_changes(self):
        media = os.path.join(self.tmp.name, 'audio.mp3')
        with open(media, 'wb') as f:
            f.write(b'\xff\xfb' + b'\x00' * 10)
        key_before = BatchState.file_key(media, base_dir=self.tmp.name)

        with open(media, 'wb') as f:
            f.write(b'\xff\xfb' + b'\x00' * 500)
        key_after = BatchState.file_key(media, base_dir=self.tmp.name)
        self.assertNotEqual(key_before, key_after)

    def test_file_key_uses_forward_slashes(self):
        nested_dir = os.path.join(self.tmp.name, 'nested')
        os.makedirs(nested_dir)
        media = os.path.join(nested_dir, 'audio.mp3')
        with open(media, 'wb') as f:
            f.write(b'\xff\xfb')
        key = BatchState.file_key(media, base_dir=self.tmp.name)
        self.assertTrue(key.startswith('nested/audio.mp3|'))
        self.assertNotIn('\\', key)

    def test_source_key_helpers(self):
        self.assertTrue(source_key_for_path(self.tmp.name).startswith('dir:'))
        self.assertEqual(
            source_key_for_url(' https://drive.google.com/drive/folders/x '),
            'url:https://drive.google.com/drive/folders/x',
        )

    def test_manifest_is_valid_json_on_disk(self):
        state = self.make_state()
        state.mark_done('a.mp3|10|1')
        with open(state.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['source'], state.source_key)
        self.assertIn('a.mp3|10|1', data['files'])


class MediaScanTest(unittest.TestCase):
    def test_recursive_scan_finds_supported_media_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, 'sub')
            os.makedirs(nested)
            keep_a = os.path.join(tmp, 'a.mp3')
            keep_b = os.path.join(nested, 'b.wav')
            skip = os.path.join(tmp, 'notes.txt')
            for path in (keep_a, keep_b):
                with open(path, 'wb') as f:
                    f.write(b'\x00')
            with open(skip, 'w', encoding='utf-8') as f:
                f.write('hello')

            found = collect_supported_files([tmp])
            self.assertEqual(sorted(os.path.basename(p) for p in found), ['a.mp3', 'b.wav'])


if __name__ == '__main__':
    unittest.main()
