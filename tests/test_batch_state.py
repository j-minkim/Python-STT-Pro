import json
import os
import sys
import tempfile
import unicodedata
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_state import CompletionIndex
from media_scan import collect_supported_files


def make_media(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\xff\xfb' + b'\x00' * 8)
    return path


class CompletionIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_dir = os.path.join(self.tmp.name, 'state')

    def tearDown(self):
        self.tmp.cleanup()

    def make_index(self, options=None):
        return CompletionIndex(options=options, state_dir=self.state_dir)

    def test_done_persists_across_instances(self):
        media = make_media(os.path.join(self.tmp.name, 'a.mp3'))
        key = CompletionIndex.file_key(media)
        index = self.make_index()
        self.assertFalse(index.is_done(key))
        index.mark_done(key, outputs={'txt': '/out/a.txt'})
        self.assertTrue(self.make_index().is_done(key))

    def test_parent_and_subfolder_submissions_share_keys(self):
        # The core incremental-transcription property: a file completed via a
        # subfolder submission is recognized when its parent is submitted.
        media = make_media(os.path.join(self.tmp.name, 'parent', 'child', 'a.mp3'))
        key_from_child = CompletionIndex.file_key(media, base_dir=os.path.join(self.tmp.name, 'parent', 'child'))
        key_from_parent = CompletionIndex.file_key(media, base_dir=os.path.join(self.tmp.name, 'parent'))
        key_absolute = CompletionIndex.file_key(media)
        self.assertEqual(key_from_child, key_from_parent)
        self.assertEqual(key_from_child, key_absolute)

        index = self.make_index()
        index.mark_done(key_from_child)
        self.assertTrue(self.make_index().is_done(key_from_parent))

    def test_failed_files_are_retried(self):
        media = make_media(os.path.join(self.tmp.name, 'a.mp3'))
        key = CompletionIndex.file_key(media)
        index = self.make_index()
        index.mark_failed(key, 'boom')
        reloaded = self.make_index()
        self.assertFalse(reloaded.is_done(key))
        self.assertEqual(reloaded.data['files'][key]['status'], 'failed')

    def test_reset_prefix_only_clears_that_folder(self):
        inside = make_media(os.path.join(self.tmp.name, 'target', 'a.mp3'))
        outside = make_media(os.path.join(self.tmp.name, 'other', 'b.mp3'))
        index = self.make_index()
        index.mark_done(CompletionIndex.file_key(inside))
        index.mark_done(CompletionIndex.file_key(outside))

        removed = index.reset_prefix(os.path.join(self.tmp.name, 'target'))
        self.assertEqual(removed, 1)
        reloaded = self.make_index()
        self.assertFalse(reloaded.is_done(CompletionIndex.file_key(inside)))
        self.assertTrue(reloaded.is_done(CompletionIndex.file_key(outside)))

    def test_reset_files_matches_relative_display_names(self):
        media = make_media(os.path.join(self.tmp.name, 'folder', 'sub', '학생상담.mp3'))
        index = self.make_index()
        index.mark_done(CompletionIndex.file_key(media))
        self.assertEqual(index.reset_files(['sub/학생상담.mp3']), 1)
        self.assertFalse(self.make_index().is_done(CompletionIndex.file_key(media)))

    def test_options_mismatch_reprocesses(self):
        media = make_media(os.path.join(self.tmp.name, 'a.mp3'))
        key = CompletionIndex.file_key(media)
        self.make_index().mark_done(key)

        diarized = self.make_index(options={'diarize': True})
        self.assertFalse(diarized.is_done(key))
        diarized.mark_done(key)

        self.assertTrue(self.make_index(options={'diarize': True}).is_done(key))
        self.assertFalse(self.make_index().is_done(key))
        self.assertFalse(self.make_index(options={'diarize': True, 'num_speakers': 2}).is_done(key))

    def test_options_key_order_does_not_matter(self):
        media = make_media(os.path.join(self.tmp.name, 'a.mp3'))
        key = CompletionIndex.file_key(media)
        self.make_index(options={'num_speakers': 2, 'diarize': True}).mark_done(key)
        self.assertTrue(self.make_index(options={'diarize': True, 'num_speakers': 2}).is_done(key))

    def test_file_key_changes_when_file_changes(self):
        media = os.path.join(self.tmp.name, 'audio.mp3')
        with open(media, 'wb') as f:
            f.write(b'\xff\xfb' + b'\x00' * 10)
        key_before = CompletionIndex.file_key(media)
        with open(media, 'wb') as f:
            f.write(b'\xff\xfb' + b'\x00' * 500)
        self.assertNotEqual(key_before, CompletionIndex.file_key(media))

    def test_keys_are_nfc_normalized(self):
        nfd_name = unicodedata.normalize('NFD', '회의녹음.mp3')
        media = make_media(os.path.join(self.tmp.name, nfd_name))
        key = CompletionIndex.file_key(media)
        path_part = key.split('|')[0]
        self.assertEqual(path_part, unicodedata.normalize('NFC', path_part))
        self.assertIn(unicodedata.normalize('NFC', '회의녹음.mp3'), path_part)

    def test_corrupt_index_starts_fresh(self):
        media = make_media(os.path.join(self.tmp.name, 'a.mp3'))
        key = CompletionIndex.file_key(media)
        index = self.make_index()
        index.mark_done(key)
        with open(index.path, 'w', encoding='utf-8') as f:
            f.write('{not valid json')
        reloaded = self.make_index()
        self.assertFalse(reloaded.is_done(key))
        reloaded.mark_done(key)
        self.assertTrue(self.make_index().is_done(key))

    def test_legacy_dir_manifest_migrates(self):
        # Old per-source manifest with relative keys is absorbed as absolute
        # keys, and the legacy file is renamed *.migrated.
        media = make_media(os.path.join(self.tmp.name, 'legacyfolder', '옛파일.mp3'))
        stat = os.stat(media)
        base = os.path.normcase(os.path.realpath(os.path.join(self.tmp.name, 'legacyfolder')))
        legacy = {
            'source': 'dir:' + base,
            'files': {
                f'옛파일.mp3|{stat.st_size}|{int(stat.st_mtime)}': {'status': 'done', 'completed_at': 1.0},
                'gdrive:abc123': {'status': 'done', 'completed_at': 1.0},
            },
        }
        os.makedirs(self.state_dir, exist_ok=True)
        legacy_path = os.path.join(self.state_dir, 'aaaa.json')
        with open(legacy_path, 'w', encoding='utf-8') as f:
            json.dump(legacy, f, ensure_ascii=False)

        index = self.make_index()
        self.assertTrue(index.is_done(CompletionIndex.file_key(media)))
        self.assertTrue(index.is_done('gdrive:abc123'))
        self.assertFalse(os.path.exists(legacy_path))
        self.assertTrue(os.path.exists(legacy_path + '.migrated'))


class MediaScanTest(unittest.TestCase):
    def test_recursive_scan_finds_supported_media_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, 'sub')
            os.makedirs(nested)
            for path in (os.path.join(tmp, 'a.mp3'), os.path.join(nested, 'b.wav')):
                with open(path, 'wb') as f:
                    f.write(b'\x00')
            with open(os.path.join(tmp, 'notes.txt'), 'w', encoding='utf-8') as f:
                f.write('hello')
            found = collect_supported_files([tmp])
            self.assertEqual(sorted(os.path.basename(p) for p in found), ['a.mp3', 'b.wav'])


if __name__ == '__main__':
    unittest.main()
