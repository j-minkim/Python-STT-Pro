import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report_summarizer import ReportSummarizer, default_chunk_chars, split_chunks

STUB = '''#!/usr/bin/env python3
import sys
args = sys.argv[1:]
out = args[args.index('--output-last-message') + 1]
data = sys.stdin.read()
with open(out, 'w', encoding='utf-8') as f:
    f.write('## 상담 개요\\n스텁 요약 (입력 %d자)' % len(data))
'''


class CodexBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stub = os.path.join(self.tmp.name, 'codex')
        with open(self.stub, 'w', encoding='utf-8') as f:
            f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IEXEC)

        home = os.path.join(self.tmp.name, 'codex_home')
        os.makedirs(home)
        open(os.path.join(home, 'auth.json'), 'w').close()

        os.environ['STT_CODEX_BIN'] = self.stub
        os.environ['STT_CODEX_HOME'] = home

    def tearDown(self):
        os.environ.pop('STT_CODEX_BIN', None)
        os.environ.pop('STT_CODEX_HOME', None)
        self.tmp.cleanup()

    def test_codex_backend_summarizes_via_cli(self):
        summarizer = ReportSummarizer(backend='codex')
        segments = [
            {'start': 0.0, 'end': 5.0, 'text': '수학 성적이 올랐습니다'},
            {'start': 5.0, 'end': 9.0, 'text': '과학 탐구를 추가하기로 했습니다'},
        ]
        report = summarizer.summarize_segments(segments, filename='상담.mp4')
        self.assertIn('# 상담 요약 — 상담.mp4', report)
        self.assertIn('스텁 요약', report)

    def test_codex_uses_large_chunks(self):
        self.assertEqual(default_chunk_chars('codex'), 120000)
        self.assertEqual(default_chunk_chars('lmstudio'), 9000)

    def test_chunk_splitting_keeps_lines(self):
        text = '\n'.join(f'[00:0{i}] 문장 {i}' for i in range(10))
        chunks = split_chunks(text, chunk_chars=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            for line in chunk.splitlines():
                self.assertTrue(line.startswith('[00:'))


if __name__ == '__main__':
    unittest.main()
