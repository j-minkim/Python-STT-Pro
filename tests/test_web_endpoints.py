"""End-to-end web API tests with a stubbed Whisper engine (no model needed).

Covers: local-folder batch through the job queue, resume manifests, event
replay, downloads with Korean filenames, QA flagging, and one-click requeue.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['SAVE_TO_DOWNLOADS'] = '0'

import batch_state
import job_store
import web_app

TIMEOUT = 15


def fake_engine(text_per_call):
    """Engine stub whose model.transcribe yields one segment with given text."""
    state = {'calls': 0}

    def transcribe(audio_path, **kwargs):
        text = text_per_call[min(state['calls'], len(text_per_call) - 1)]
        state['calls'] += 1
        seg = SimpleNamespace(
            start=0.0, end=2.0, text=text,
            words=[SimpleNamespace(start=0.0, end=2.0, word=text.split()[0])],
        )
        info = SimpleNamespace(language='ko', language_probability=0.99, duration=2.0)
        return iter([seg]), info

    return SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))


class WebEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        web_app.app.config['UPLOAD_FOLDER'] = os.path.join(root, 'uploads')
        web_app.app.config['OUTPUT_FOLDER'] = os.path.join(root, 'outputs')
        os.makedirs(web_app.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(web_app.app.config['OUTPUT_FOLDER'], exist_ok=True)
        job_store.JOBS_DIR = os.path.join(root, 'jobs')
        batch_state.STATE_DIR = os.path.join(root, 'batch_state')
        cls.client = web_app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self._orig_load_engine = web_app.load_engine

    def tearDown(self):
        web_app.load_engine = self._orig_load_engine

    def make_folder(self, *names):
        folder = tempfile.mkdtemp(dir=self.tmp.name)
        for name in names:
            with open(os.path.join(folder, name), 'wb') as f:
                f.write(b'\xff\xfb' + b'\x00' * 16)
        return folder

    def submit_folder(self, folder, **extra):
        data = {'model': 'tiny', 'language': 'ko', 'local_folder_path': folder}
        data.update(extra)
        res = self.client.post('/api/transcribe', data=data)
        self.assertEqual(res.status_code, 200, res.get_json())
        return res.get_json()['job_id']

    def wait_done(self, job_id):
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            job = web_app.jobs.get(job_id) or {}
            if job.get('status') in ('done', 'error', 'cancelled'):
                return job
            time.sleep(0.1)
        self.fail(f'job {job_id} did not finish: {web_app.jobs.get(job_id, {}).get("status")}')

    def wait_status(self, job_id, status):
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            if (web_app.jobs.get(job_id) or {}).get('status') == status:
                return
            time.sleep(0.05)
        self.fail(f'job {job_id} never reached {status}')

    def test_local_folder_batch_full_flow(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['안녕하세요 전사 테스트'])
        folder = self.make_folder('회의녹음 1회차.mp3', '회의녹음 2회차.mp3')

        job_id = self.submit_folder(folder)
        job = self.wait_done(job_id)
        self.assertEqual(job['status'], 'done', job.get('error'))
        self.assertEqual(job['batch_summary']['successful'], 2)

        # Korean filenames survive into outputs and downloads.
        stems = {f['outputs']['stem'] for f in job['files']}
        self.assertEqual(stems, {'회의녹음 1회차', '회의녹음 2회차'})
        res = self.client.get(f'/api/download/{job_id}/file/1/txt')
        self.assertEqual(res.status_code, 200)

        # Jobs list includes it; event history replays to 'done'.
        listing = self.client.get('/api/jobs').get_json()
        self.assertIn(job_id, [j['job_id'] for j in listing['jobs']])
        stream = self.client.get(f'/api/stream/{job_id}').get_data(as_text=True)
        self.assertIn('"done"', stream)
        self.assertIn('file_done', stream)

        # Persisted record exists for restart survival.
        self.assertIsNotNone(job_store.load_job(job_id))

    def test_resume_skips_completed_files(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['정상 전사 내용입니다'])
        folder = self.make_folder('a.mp3', 'b.mp3')

        first = self.wait_done(self.submit_folder(folder))
        self.assertEqual(first['batch_summary']['successful'], 2)

        second = self.wait_done(self.submit_folder(folder))
        self.assertEqual(second['batch_summary']['successful'], 0)
        self.assertEqual(second['batch_summary']['skipped'], 2)

    def test_qa_flags_hallucination_and_requeue_reprocesses(self):
        hallucinated = '네. ' * 30
        # One shared engine so the call counter spans both the original job
        # and the requeue (per-job engines would replay the hallucination).
        engine = fake_engine([hallucinated, '정상입니다', '재전사 결과입니다'])
        web_app.load_engine = lambda job_id, model: engine
        folder = self.make_folder('bad.mp3', 'good.mp3')

        job_id = self.submit_folder(folder)
        job = self.wait_done(job_id)
        flagged = [f['filename'] for f in job['qa']['flagged']]
        self.assertEqual(flagged, ['bad.mp3'])

        res = self.client.post(f'/api/requeue/{job_id}', json={'only_flagged': True})
        self.assertEqual(res.status_code, 200, res.get_json())
        body = res.get_json()
        self.assertEqual(body['reset_files'], 1)

        rerun = self.wait_done(body['job_id'])
        self.assertEqual(rerun['status'], 'done')
        self.assertEqual(rerun['batch_summary']['successful'], 1)  # only bad.mp3
        self.assertEqual(rerun['batch_summary']['skipped'], 1)
        self.assertEqual(len((rerun.get('qa') or {}).get('flagged') or []), 0)

    def test_queue_runs_jobs_sequentially(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['큐 테스트 문장'])
        first = self.submit_folder(self.make_folder('q1.mp3'))
        second = self.submit_folder(self.make_folder('q2.mp3'))
        self.assertEqual(self.wait_done(first)['status'], 'done')
        self.assertEqual(self.wait_done(second)['status'], 'done')

    def test_cancel_running_and_queued_jobs(self):
        def slow_engine(job_id, model):
            def transcribe(path, **kwargs):
                def gen():
                    for i in range(200):
                        time.sleep(0.05)
                        yield SimpleNamespace(
                            start=float(i), end=i + 1.0, text=f'느린 세그먼트 {i}',
                            words=[SimpleNamespace(start=float(i), end=i + 1.0, word='세그먼트')],
                        )
                info = SimpleNamespace(language='ko', language_probability=0.99, duration=200.0)
                return gen(), info
            return SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))

        web_app.load_engine = slow_engine
        running = self.submit_folder(self.make_folder('slow.mp3'))
        queued = self.submit_folder(self.make_folder('waiting.mp3'))

        self.wait_status(running, 'transcribing')

        # Queued job cancels instantly and never runs.
        res = self.client.post(f'/api/cancel/{queued}')
        self.assertEqual(res.get_json()['status'], 'cancelled')
        self.assertEqual(web_app.jobs[queued]['status'], 'cancelled')

        # Running job cancels cooperatively mid-file.
        res = self.client.post(f'/api/cancel/{running}')
        self.assertEqual(res.get_json()['status'], 'cancelling')
        job = self.wait_done(running)
        self.assertEqual(job['status'], 'cancelled')
        self.assertIsNone(web_app.jobs[queued].get('batch_summary'))

        # Cancelling a finished job is rejected.
        res = self.client.post(f'/api/cancel/{running}')
        self.assertEqual(res.status_code, 400)

    def test_duplicate_folder_submission_rejected(self):
        def slow_engine(job_id, model):
            def transcribe(path, **kwargs):
                def gen():
                    for i in range(100):
                        time.sleep(0.05)
                        yield SimpleNamespace(start=float(i), end=i + 1.0, text='중복 테스트',
                                              words=[SimpleNamespace(start=float(i), end=i + 1.0, word='중복')])
                return gen(), SimpleNamespace(language='ko', language_probability=0.99, duration=100.0)
            return SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))

        web_app.load_engine = slow_engine
        folder = self.make_folder('dup.mp3')
        first = self.submit_folder(folder)
        res = self.client.post('/api/transcribe', data={
            'model': 'tiny', 'language': 'ko', 'local_folder_path': folder,
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn('이미 진행 중', res.get_json()['error'])

        self.client.post(f'/api/cancel/{first}')
        self.wait_done(first)

    def test_invalid_folder_rejected(self):
        res = self.client.post('/api/transcribe', data={
            'model': 'tiny', 'local_folder_path': '/no/such/folder/anywhere',
        })
        self.assertEqual(res.status_code, 400)


if __name__ == '__main__':
    unittest.main()
