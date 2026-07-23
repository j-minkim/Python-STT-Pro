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

    def test_parent_folder_submission_skips_subfolder_completions(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['증분 전사 테스트'])
        parent = tempfile.mkdtemp(dir=self.tmp.name)
        child_a = os.path.join(parent, '9월_고1')
        child_b = os.path.join(parent, '9월_예비고1')
        os.makedirs(child_a); os.makedirs(child_b)
        for path in (os.path.join(child_a, 'a.mp3'), os.path.join(child_b, 'b.mp3')):
            with open(path, 'wb') as f:
                f.write(b'\xff\xfb' + b'\x00' * 16)

        # Transcribe one subfolder, then submit the PARENT: only the new
        # file in the other subfolder should be processed.
        first = self.wait_done(self.submit_folder(child_a))
        self.assertEqual(first['batch_summary']['successful'], 1)

        second = self.wait_done(self.submit_folder(parent))
        self.assertEqual(second['batch_summary']['skipped'], 1)
        self.assertEqual(second['batch_summary']['successful'], 1)

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

    def test_downloads_subfolder_rule(self):
        base = os.path.join(self.tmp.name, '컨설팅 영상_2025', '9월_고1')
        os.makedirs(base, exist_ok=True)
        media = os.path.join(base, 'x.mp4')
        open(media, 'wb').close()
        self.assertEqual(web_app.downloads_subfolder_for(media), '2025년_9월_고1')

        yeared = os.path.join(self.tmp.name, '자료실 2026')
        os.makedirs(yeared, exist_ok=True)
        media2 = os.path.join(yeared, 'y.mp4')
        open(media2, 'wb').close()
        self.assertEqual(web_app.downloads_subfolder_for(media2), '자료실 2026')

    def test_downloads_auto_organized_per_source_folder(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['자동 정리 테스트'])
        downloads = tempfile.mkdtemp(dir=self.tmp.name)
        parent = os.path.join(self.tmp.name, '컨설팅 영상_2025', '9월_고2')
        os.makedirs(parent, exist_ok=True)
        with open(os.path.join(parent, '상담녹화.mp3'), 'wb') as f:
            f.write(b'\xff\xfb' + b'\x00' * 16)

        os.environ['SAVE_TO_DOWNLOADS'] = '1'
        os.environ['STT_DOWNLOADS_DIR'] = downloads
        try:
            job = self.wait_done(self.submit_folder(parent))
        finally:
            os.environ['SAVE_TO_DOWNLOADS'] = '0'
            os.environ.pop('STT_DOWNLOADS_DIR', None)

        self.assertEqual(job['status'], 'done', job.get('error'))
        organized = os.path.join(downloads, '2025년_9월_고2', '상담녹화.txt')
        self.assertTrue(os.path.exists(organized), os.listdir(downloads))

    def test_summary_report_generated(self):
        import report_summarizer

        class FakeSummarizer:
            def __init__(self, backend=None, model=None):
                pass

            def summarize_segments(self, segments, filename=''):
                return f'# 상담 요약 — {filename}\n\n## 상담 개요\n요약 본문입니다.'

        original = report_summarizer.ReportSummarizer
        report_summarizer.ReportSummarizer = FakeSummarizer
        web_app.load_engine = lambda job_id, model: fake_engine(['요약 대상 전사입니다'])
        try:
            folder = self.make_folder('요약대상.mp3')
            job_id = self.submit_folder(folder, summary='1', summary_backend='openai')
            job = self.wait_done(job_id)
        finally:
            report_summarizer.ReportSummarizer = original

        self.assertEqual(job['status'], 'done', job.get('error'))
        outputs = job['files'][0]['outputs']
        self.assertIn('summary', outputs)
        with open(outputs['summary'], encoding='utf-8') as f:
            self.assertIn('상담 요약', f.read())
        self.assertIn('summary', job['files'][0]['download_urls'])
        res = self.client.get(job['files'][0]['download_urls']['summary'])
        self.assertEqual(res.status_code, 200)

    def test_search_finds_segments(self):
        web_app.load_engine = lambda job_id, model: fake_engine(['등대지기 프로젝트 이야기입니다'])
        self.wait_done(self.submit_folder(self.make_folder('검색용.mp3')))

        res = self.client.get('/api/search?q=등대지기')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['results'][0]['filename'], '검색용.mp3')
        self.assertIn('등대지기', data['results'][0]['text'])

        self.assertEqual(self.client.get('/api/search?q=ㄱ').status_code, 400)

    def test_batch_aborts_when_source_folder_disappears(self):
        import shutil

        def slow_engine(job_id, model):
            def transcribe(path, **kwargs):
                def gen():
                    time.sleep(0.5)
                    yield SimpleNamespace(start=0.0, end=1.0, text='연결 끊김 테스트',
                                          words=[SimpleNamespace(start=0.0, end=1.0, word='테스트')])
                return gen(), SimpleNamespace(language='ko', language_probability=0.99, duration=1.0)
            return SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))

        web_app.load_engine = slow_engine
        folder = self.make_folder('f1.mp3', 'f2.mp3', 'f3.mp3')
        os.environ['STT_RECONNECT_WAIT_MINUTES'] = '0'  # abort immediately
        try:
            job_id = self.submit_folder(folder)
            self.wait_status(job_id, 'transcribing')
            shutil.rmtree(folder)  # simulate the share unmounting mid-batch
            job = self.wait_done(job_id)
        finally:
            os.environ.pop('STT_RECONNECT_WAIT_MINUTES', None)

        self.assertEqual(job['status'], 'error')
        self.assertIn('연결이 끊겼습니다', job['error'] or '')
        # It aborted once instead of failing every remaining file.
        self.assertEqual((job.get('batch_summary') or {}).get('failed'), 0)

    def test_batch_resumes_when_source_folder_reconnects(self):
        import shutil

        def slow_engine(job_id, model):
            def transcribe(path, **kwargs):
                def gen():
                    time.sleep(0.4)
                    yield SimpleNamespace(start=0.0, end=1.0, text='재연결 테스트',
                                          words=[SimpleNamespace(start=0.0, end=1.0, word='테스트')])
                return gen(), SimpleNamespace(language='ko', language_probability=0.99, duration=1.0)
            return SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))

        web_app.load_engine = slow_engine
        folder = self.make_folder('r1.mp3', 'r2.mp3')
        os.environ['STT_RECONNECT_WAIT_MINUTES'] = '0.2'   # up to 12s
        os.environ['STT_RECONNECT_POLL_SECONDS'] = '0.2'
        try:
            job_id = self.submit_folder(folder)
            self.wait_status(job_id, 'transcribing')
            backup = folder + '_backup'
            shutil.copytree(folder, backup)
            shutil.rmtree(folder)          # share drops during file 1
            time.sleep(1.0)
            shutil.move(backup, folder)    # ...and comes back

            job = self.wait_done(job_id)
        finally:
            os.environ.pop('STT_RECONNECT_WAIT_MINUTES', None)
            os.environ.pop('STT_RECONNECT_POLL_SECONDS', None)

        self.assertEqual(job['status'], 'done', job.get('error'))
        self.assertEqual(job['batch_summary']['successful'], 2)

    def test_invalid_folder_rejected(self):
        res = self.client.post('/api/transcribe', data={
            'model': 'tiny', 'local_folder_path': '/no/such/folder/anywhere',
        })
        self.assertEqual(res.status_code, 400)


if __name__ == '__main__':
    unittest.main()
