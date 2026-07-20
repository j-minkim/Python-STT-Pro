"""Per-source batch progress tracking so interrupted runs can resume.

Each batch source (a local folder, a Google Drive folder URL, or a CLI list
file) gets one JSON manifest under data/batch_state/. The manifest is
rewritten atomically after every file, so a crash loses at most the file that
was in flight. Completed files are skipped on the next run; failed files are
retried.
"""

import hashlib
import json
import os
import time

STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'batch_state')


def source_key_for_path(path):
    return 'dir:' + os.path.normcase(os.path.realpath(path))


def source_key_for_list(path):
    return 'list:' + os.path.normcase(os.path.realpath(path))


def source_key_for_url(url):
    return 'url:' + url.strip()


class BatchState:
    def __init__(self, source_key, state_dir=None):
        self.source_key = source_key
        directory = state_dir or STATE_DIR
        os.makedirs(directory, exist_ok=True)
        digest = hashlib.sha1(source_key.encode('utf-8')).hexdigest()[:16]
        self.path = os.path.join(directory, f'{digest}.json')
        self.data = self._load()

    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('source') != self.source_key or not isinstance(data.get('files'), dict):
                raise ValueError('stale or corrupt state')
            return data
        except (OSError, ValueError, json.JSONDecodeError):
            return {'source': self.source_key, 'created_at': time.time(), 'files': {}}

    @staticmethod
    def file_key(path, base_dir=None):
        """Identity for a local file: relative path + size + mtime.

        A changed source file therefore gets a new key and is transcribed
        again. Keys use '/' separators and normcase so manifests written on
        Windows and macOS agree.
        """
        if base_dir:
            rel = os.path.relpath(path, base_dir)
        else:
            rel = os.path.basename(path)
        rel = os.path.normcase(rel).replace(os.sep, '/')
        try:
            stat = os.stat(path)
            return f'{rel}|{stat.st_size}|{int(stat.st_mtime)}'
        except OSError:
            return f'{rel}|unknown'

    def is_done(self, key):
        return self.data['files'].get(key, {}).get('status') == 'done'

    def mark_done(self, key, outputs=None):
        entry = {'status': 'done', 'completed_at': time.time()}
        if outputs:
            entry['outputs'] = outputs
        self.data['files'][key] = entry
        self._save()

    def mark_failed(self, key, error):
        self.data['files'][key] = {
            'status': 'failed',
            'failed_at': time.time(),
            'error': str(error),
        }
        self._save()

    def reset(self):
        self.data['files'] = {}
        self._save()

    def _save(self):
        self.data['updated_at'] = time.time()
        tmp_path = self.path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
