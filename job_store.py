"""Persistent job registry: one JSON per job under data/jobs/.

Keeps job metadata (status, file list, download URLs, QA report, requeue
source) across server restarts. Segment bodies are intentionally not
persisted — transcripts live in data/outputs/.
"""

import glob
import json
import os
import time

from runtime_config import DATA_ROOT

JOBS_DIR = os.path.join(DATA_ROOT, 'jobs')

# Statuses that mean "still working" — anything found in one of these states
# at server startup did not survive the previous process.
ACTIVE_STATUSES = ('queued', 'loading', 'downloading', 'transcribing', 'diarizing', 'batch')

# Fields copied from the in-memory job dict into the persisted record.
PERSISTED_FIELDS = (
    'status', 'filename', 'created_at', 'error', 'outputs', 'failures',
    'source', 'params', 'batch_summary', 'qa', 'expired',
    'subtitle', 'translation',
)


def _job_path(job_id):
    return os.path.join(JOBS_DIR, f'{job_id}.json')


def save_job(job_id, job):
    """Persist the durable subset of an in-memory job dict (atomic)."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    record = {'job_id': job_id, 'updated_at': time.time()}
    for field in PERSISTED_FIELDS:
        if job.get(field) is not None:
            record[field] = job[field]
    files = job.get('files')
    if files:
        record['files'] = [
            {
                'filename': f.get('filename'),
                'outputs': f.get('outputs'),
                'download_urls': f.get('download_urls'),
                'translation_urls': f.get('translation_urls'),
                'segment_count': len(f.get('segments') or []),
            }
            for f in files
        ]
    tmp = _job_path(job_id) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _job_path(job_id))


def load_job(job_id):
    try:
        with open(_job_path(job_id), encoding='utf-8') as f:
            record = json.load(f)
        return record if isinstance(record, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs(limit=50):
    """Newest-first summaries of persisted jobs."""
    records = []
    for path in glob.glob(os.path.join(JOBS_DIR, '*.json')):
        try:
            with open(path, encoding='utf-8') as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get('job_id'):
            records.append(record)
    records.sort(key=lambda r: r.get('created_at') or 0, reverse=True)
    return records[:limit]


def mark_interrupted_jobs():
    """Flag jobs left in an active state by a dead process. Returns count."""
    changed = 0
    for record in list_jobs(limit=1000):
        if record.get('status') in ACTIVE_STATUSES:
            record['status'] = 'interrupted'
            record['updated_at'] = time.time()
            tmp = _job_path(record['job_id']) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _job_path(record['job_id']))
            changed += 1
    return changed


def mark_expired(job_id):
    record = load_job(job_id)
    if record:
        record['expired'] = True
        record['updated_at'] = time.time()
        tmp = _job_path(job_id) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _job_path(job_id))
