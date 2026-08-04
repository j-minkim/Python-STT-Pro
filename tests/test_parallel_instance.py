import json
import os
import subprocess
import sys
import unittest


class ParallelInstanceTest(unittest.TestCase):
    def test_named_instance_uses_isolated_data_directories(self):
        # Given: a second web instance has its own stable instance name.
        env = os.environ.copy()
        env['STT_INSTANCE'] = 'shared'

        # When: the application modules resolve their runtime directories.
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import json, batch_state, job_store, web_app; '
                    'print(json.dumps({'
                    '"uploads": web_app.app.config["UPLOAD_FOLDER"], '
                    '"outputs": web_app.app.config["OUTPUT_FOLDER"], '
                    '"jobs": job_store.JOBS_DIR, '
                    '"state": batch_state.STATE_DIR'
                    '}))'
                ),
            ],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        paths = json.loads(result.stdout)

        # Then: job data is isolated while the completion index stays shared.
        instance_root = os.path.join('data', 'instances', 'shared')
        self.assertTrue(all(instance_root in paths[key] for key in ('uploads', 'outputs', 'jobs')), paths)
        self.assertNotIn(instance_root, paths['state'])
        self.assertIn(os.path.join('data', 'batch_state'), paths['state'])

    def test_stale_completion_index_writers_merge_updates(self):
        from batch_state import CompletionIndex
        import tempfile

        # Given: two server processes loaded the same completion index snapshot.
        with tempfile.TemporaryDirectory() as state_dir:
            first = CompletionIndex(state_dir=state_dir)
            second = CompletionIndex(state_dir=state_dir)

            # When: each process records a different completed file.
            first.mark_done('file:first')
            second.mark_done('file:second')

            # Then: neither completion record overwrites the other.
            combined = CompletionIndex(state_dir=state_dir)
            self.assertTrue(combined.is_done('file:first'))
            self.assertTrue(combined.is_done('file:second'))


if __name__ == '__main__':
    unittest.main()
