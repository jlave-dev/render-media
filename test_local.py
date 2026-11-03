"""Exercise the single-machine recorder transport without a ComfyUI server."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class LocalTransportTest(unittest.TestCase):
    def test_create_show_and_copy_local_input(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, HOME=home)
            graph = Path(home) / 'graph.json'
            graph.write_text(json.dumps({'1': {'class_type': 'LoadImage', 'inputs': {'image': 'placeholder.png'}},
                                         '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}}))
            image = Path(home) / 'input.png'
            image.write_bytes(b'local input')
            # Create and transfer through the CLI, then stop before preparation by
            # checking the archived input after the expected missing-backend failure.
            proc = subprocess.run([str(ROOT / 'render.py'), '--local', 'submit', str(graph), '--project', 'test',
                                   '--title', 'Local input', '--input', f'1.image={image}', '--prepare-only'],
                                  env=env, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('Created ', proc.stderr)
            run_id = proc.stderr.split('Created ', 1)[1].splitlines()[0]
            show = subprocess.run([str(ROOT / 'render.py'), '--local', 'show', run_id], env=env,
                                  text=True, capture_output=True, check=True)
            manifest = json.loads(show.stdout)
            self.assertEqual((Path(manifest['path']) / 'inputs/0.png').read_bytes(), b'local input')


if __name__ == '__main__':
    unittest.main()
