"""Global file-completion index so interrupted or repeated runs can resume.

One JSON registry (data/batch_state/global_index.json) records every media
file ever completed, keyed by absolute path + size + mtime. Because the key
does not depend on which folder was submitted, transcribing a subfolder and
later submitting its parent folder skips the already-done files — parent
submissions become incremental runs.

The registry is rewritten atomically after every file, so a crash loses at
most the file that was in flight. Completed files are skipped on the next
run (only when processing options match); failed files are retried; changed
source files (size/mtime) are re-processed.

Legacy per-source manifests (dir:/url: sources with relative keys) are
absorbed into the global index on first load and preserved as *.migrated.
"""

import glob
import hashlib  # noqa: F401  (kept for external imports)
import json
import os
import time
import unicodedata

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'batch_state')
GLOBAL_INDEX_NAME = 'global_index.json'


def _nfc(text):
    return unicodedata.normalize('NFC', text or '')


def _norm_path(path):
    """Canonical absolute path form used inside keys ('/'-separated, NFC)."""
    return _nfc(os.path.normcase(os.path.realpath(path))).replace(os.sep, '/')


# Option keys that never define a file's identity for resume purposes — they
# don't change which output files are produced. num_speakers is a diarization
# hint (blank vs "2" yields the same _diarized.* outputs), so records written
# with or without it must still match. Dropping them here also migrates older
# index entries that stored num_speakers without any data rewrite.
_NON_IDENTITY_OPTION_KEYS = {'num_speakers'}


def _normalize_options(options):
    if not options:
        return {}
    return {
        key: options[key]
        for key in sorted(options)
        if key not in _NON_IDENTITY_OPTION_KEYS
    }


class CompletionIndex:
    def __init__(self, options=None, state_dir=None):
        """options: processing options (e.g. {'diarize': True}) that must match
        a completed entry for it to be skipped — rerunning with different
        options reprocesses the file."""
        self.options = _normalize_options(options)
        self.dir = state_dir or STATE_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, GLOBAL_INDEX_NAME)
        self.data = self._load()
        self._migrate_legacy()

    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data.get('files'), dict):
                raise ValueError('corrupt index')
            data['files'] = {
                _nfc(key): value for key, value in data['files'].items()
            }
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return {'version': 2, 'created_at': time.time(), 'files': {}}

    # ---- keys ----------------------------------------------------------

    @staticmethod
    def file_key(path, base_dir=None):
        """Identity of a local file: absolute path + size + mtime.

        base_dir is accepted for backward compatibility but ignored — keys
        are path-absolute so overlapping folder submissions agree.
        """
        abs_path = _norm_path(path)
        try:
            stat = os.stat(path)
            return f'{abs_path}|{stat.st_size}|{int(stat.st_mtime)}'
        except OSError:
            return f'{abs_path}|unknown'

    # ---- queries / updates ---------------------------------------------

    def is_done(self, key):
        entry = self.data['files'].get(_nfc(key))
        if not entry or entry.get('status') != 'done':
            return False
        return _normalize_options(entry.get('options')) == self.options

    def mark_done(self, key, outputs=None):
        entry = {'status': 'done', 'completed_at': time.time()}
        if self.options:
            entry['options'] = self.options
        if outputs:
            entry['outputs'] = outputs
        self.data['files'][_nfc(key)] = entry
        self._save()

    def mark_failed(self, key, error):
        self.data['files'][_nfc(key)] = {
            'status': 'failed',
            'failed_at': time.time(),
            'error': str(error),
        }
        self._save()

    def reset_prefix(self, folder):
        """Drop every record under a folder ("완료 기록 무시" / --fresh)."""
        prefix = _norm_path(folder) + '/'
        removed = [
            key for key in self.data['files']
            if key.split('|')[0].startswith(prefix)
        ]
        for key in removed:
            del self.data['files'][key]
        if removed:
            self._save()
        return len(removed)

    def reset_files(self, display_names):
        """Drop records whose path part ends with any display name (relative
        path or bare filename, e.g. QA-flagged files)."""
        targets = {
            _nfc(os.path.normcase(name)).replace(os.sep, '/')
            for name in display_names
        }
        removed = []
        for key in self.data['files']:
            path_part = key.split('|')[0]
            for target in targets:
                if path_part == target or path_part.endswith('/' + target):
                    removed.append(key)
                    break
        for key in removed:
            del self.data['files'][key]
        if removed:
            self._save()
        return len(removed)

    def _save(self):
        self.data['updated_at'] = time.time()
        tmp_path = self.path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    # ---- legacy migration ----------------------------------------------

    def _migrate_legacy(self):
        """Absorb old per-source manifests (relative keys) once."""
        migrated_any = False
        for mpath in glob.glob(os.path.join(self.dir, '*.json')):
            if os.path.basename(mpath) == GLOBAL_INDEX_NAME:
                continue
            try:
                with open(mpath, encoding='utf-8') as f:
                    legacy = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            source = legacy.get('source') or ''
            files = legacy.get('files')
            if not isinstance(files, dict):
                continue

            if source.startswith('dir:'):
                base = _nfc(source[4:]).replace(os.sep, '/').rstrip('/')
                for key, entry in files.items():
                    parts = _nfc(key).split('|')
                    rel = parts[0]
                    if rel.startswith('gdrive'):
                        new_key = _nfc(key)
                    else:
                        new_key = '|'.join([f'{base}/{rel}'] + parts[1:])
                    self.data['files'].setdefault(new_key, entry)
            elif source.startswith('url:'):
                for key, entry in files.items():
                    if key.split('|')[0].startswith('gdrive'):
                        self.data['files'].setdefault(_nfc(key), entry)
            # 'list:' sources: relative keys can't be resolved to absolute
            # paths reliably — those files will simply re-transcribe.

            os.replace(mpath, mpath + '.migrated')
            migrated_any = True

        if migrated_any:
            self._save()


# Backward-compatible aliases: older call sites constructed BatchState with a
# source key; the global index no longer needs one.
BatchState = CompletionIndex


def source_key_for_path(path):
    return 'dir:' + os.path.normcase(os.path.realpath(path))


def source_key_for_list(path):
    return 'list:' + os.path.normcase(os.path.realpath(path))


def source_key_for_url(url):
    return 'url:' + url.strip()
