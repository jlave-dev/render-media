import base64
import copy
import hashlib
import json
import socket
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import recorder as r
import cleanup
import library
import orphan_cleanup
import render
from common import digest, read, write

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j3ioAAAAASUVORK5CYII=')


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive'
        self.input = self.root / 'input'
        self.output = self.root / 'output'
        self.code = self.root / 'code'
        self.models = self.root / 'models'
        for p in [self.input, self.output, self.code, self.models]:
            p.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.code)], check=True)
        (self.code / 'main.py').write_text('print("fixture")\n')
        self.model = self.models / 'model.safetensors'
        self.model.write_bytes(b'checkpoint fixture')
        self.proc = {'pid': 123, 'argv': ['python', 'main.py'], 'cwd': str(self.code), 'flags': [],
                     'environment': {}, 'input': str(self.input), 'output': str(self.output), 'temp': str(self.output)}
        self.snapshot = {'process': self.proc, 'model_paths': {'diffusion_models': [str(self.models)]},
                         'packages': [], 'python': 'test', 'driver': 'test'}
        self.history, self.submissions, self.busy, self.lose_ack = {}, 0, False, False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value, code=200):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                if self.path == '/object_info':
                    return self.reply({
                        'SaveImage': {'input': {'required': {}}},
                        'LoadImage': {'input': {'required': {'image': [['ref.png'], {'image_upload': True}]}}},
                        'Loader': {'input': {'required': {}}},
                        'Wildcard': {'input': {'required': {}}},
                        'RandomNoise': {'input': {'required': {'noise_seed': ['INT', {}]}}},
                    })
                if self.path == '/queue':
                    return self.reply({'queue_running': [[1, 'someone-else']] if fixture.busy else [], 'queue_pending': []})
                if self.path.startswith('/history/'):
                    key = self.path.removeprefix('/history/')
                    return self.reply({key: fixture.history[key]} if key in fixture.history else {})
                if self.path.startswith('/history?'):
                    return self.reply(fixture.history)
                self.reply({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fixture.submissions += 1
                prefix = body['prompt']['2']['inputs']['filename_prefix']
                target = fixture.output / (prefix + '_00001.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(PNG)
                fixture.history['prompt-1'] = {'prompt': [0, 'prompt-1', body['prompt'], body['extra_data']],
                                              'status': {'status_str': 'success'},
                                              'outputs': {'2': {'images': [{'filename': target.name, 'subfolder': str(target.parent.relative_to(fixture.output)), 'type': 'output'}]}}}
                if fixture.lose_ack:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.reply({'prompt_id': 'prompt-1'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.patches = [patch.object(r, 'ROOT', self.archive), patch.dict(r.PORTS, {'comfy': self.server.server_port}),
                        patch.object(r, 'runtime', side_effect=lambda _: copy.deepcopy(self.snapshot)),
                        patch.object(r, 'process', return_value=self.proc)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def create(self, graph=None, **kw):
        graph = graph or {'1': {'class_type': 'Loader', 'inputs': {'model': 'model.safetensors'}},
                          '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': '/untrusted/old/path'}}}
        result = r.create({'project': 'test-project', 'title': 'Fixture', 'recipe': 'test', 'backend': 'comfy',
                           'graph': graph, **kw})
        return result['run_id'], r.run_path(result['run_id'])

    def test_submit_archives_actual_graph_assets_inputs_and_outputs(self):
        (self.input / 'ref.png').write_bytes(PNG)
        run_id, path = self.create(graph={
            '1': {'class_type': 'Loader', 'inputs': {'model': 'model.safetensors'}},
            '3': {'class_type': 'LoadImage', 'inputs': {'image': 'ref.png'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': '../../bad'}}})
        r.execute(run_id)
        m = read(path / 'manifest.json')
        self.assertEqual(m['state'], 'succeeded', m.get('error'))
        self.assertEqual(m['assets'][0]['sha256'], digest(self.model))
        self.assertEqual(len(m['inputs']), 1)
        self.assertEqual(Path(m['outputs'][0]['path']).read_bytes(), PNG)
        self.assertTrue(read(path / 'graph-api.json')['2']['inputs']['filename_prefix'].startswith('media-runs/' + run_id))
        self.assertTrue(r.verify(path)['ok'])
        trace = (path / 'memory-samples.csv').read_text().splitlines()
        self.assertGreaterEqual(len(trace), 2)
        self.assertIn('available_kib', trace[0])
        self.assertGreater(r.dispatch({'action': 'memory', 'run_id': run_id})['min_available_kib'], 0)
        self.assertEqual(self.submissions, 1)
        r.execute(run_id)
        self.assertEqual(read(path / 'manifest.json')['state'], 'succeeded')
        self.assertEqual(self.submissions, 1)

    def test_busy_backend_is_not_submitted(self):
        self.busy = True
        run_id, path = self.create()
        r.execute(run_id)
        self.assertEqual(read(path / 'manifest.json')['state'], 'failed')
        self.assertEqual(self.submissions, 0)

    def test_explicit_busy_append(self):
        self.busy = True
        run_id, path = self.create(allow_busy=True)
        r.execute(run_id)
        self.assertEqual(read(path / 'manifest.json')['state'], 'succeeded')

    def test_lost_ack_reconciles_without_reposting(self):
        self.lose_ack = True
        run_id, path = self.create()
        r.execute(run_id)
        self.assertEqual(read(path / 'manifest.json')['state'], 'uncertain')
        r.execute(run_id, recover=True)
        self.assertEqual(read(path / 'manifest.json')['state'], 'succeeded')
        self.assertEqual(self.submissions, 1)

    def test_missing_history_never_reposts(self):
        run_id, path = self.create()
        m = read(path / 'manifest.json')
        r.save(path, m, 'submitting')
        r.execute(run_id, recover=True)
        self.assertEqual(read(path / 'manifest.json')['state'], 'uncertain')
        self.assertEqual(self.submissions, 0)

    def test_hash_drift_is_detected_even_when_size_and_mtime_match(self):
        old = r.asset_record(self.model)
        self.model.write_bytes(b'changed checkpoint')
        self.assertNotEqual(old['sha256'], r.asset_record(self.model)['sha256'])
        run_id, path = self.create()
        r.execute(run_id)
        self.model.write_bytes(b'drift')
        self.assertFalse(r.verify(path)['ok'])

    def test_dynamic_and_unresolved_inputs_fail_before_submission(self):
        run_id, path = self.create(graph={
            '1': {'class_type': 'Wildcard', 'inputs': {'model': 'missing.gguf'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}})
        r.execute(run_id)
        m = read(path / 'manifest.json')
        self.assertEqual(m['state'], 'failed')
        self.assertEqual(m['provenance'], 'incomplete')
        self.assertEqual(self.submissions, 0)

    def test_random_noise_requires_a_literal_seed_for_complete_provenance(self):
        for seed, complete in [(42, True), (0, True), (2**64 - 1, True),
                               (['3', 0], False), (None, False), (True, False), (-1, False), (2**64, False)]:
            with self.subTest(seed=seed):
                graph = {'2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}},
                         '8': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}}}
                run_id, path = self.create(graph=graph, allow_incomplete=True)
                m = r.prepare(path, read(path / 'manifest.json'))
                self.assertEqual(not m['gaps'], complete, m['gaps'])
                self.assertEqual(read(path / 'graph-api.json')['8']['inputs']['noise_seed'], seed)

    def test_path_traversal_and_invalid_run_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            r.contained(self.input, '../../etc/passwd')
        with self.assertRaises(ValueError):
            r.run_path('../../something')

    def test_upload_digest_is_checked(self):
        run_id, path = self.create(graph={
            '1': {'class_type': 'LoadImage', 'inputs': {'image': '__UPLOAD__/0.png'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}},
            uploads=[{'binding': '1.image', 'name': '0.png', 'sha256': 'wrong'}])
        (path / 'inputs' / '0.png').write_bytes(PNG)
        r.execute(run_id)
        self.assertEqual(read(path / 'manifest.json')['state'], 'failed')
        self.assertEqual(self.submissions, 0)

    def test_input_dedupe_ignores_filename(self):
        a, b = self.root / 'first.png', self.root / 'different-name.png'
        a.write_bytes(PNG)
        b.write_bytes(PNG)
        self.assertEqual(r.dedupe_input(a), r.dedupe_input(b))
        self.assertEqual(a.stat().st_ino, b.stat().st_ino)
        self.assertEqual(a.read_bytes(), PNG)

    def test_input_dedupe_refuses_corrupt_existing_blob(self):
        source = self.root / 'reference.png'
        source.write_bytes(PNG)
        blobs = self.archive / 'blobs' / 'inputs'
        blobs.mkdir(parents=True)
        (blobs / hashlib.sha256(PNG).hexdigest()).write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            r.dedupe_input(source)
        self.assertEqual(source.read_bytes(), PNG)

    def test_staging_cleanup_checks_archive_and_queue(self):
        (self.input / 'ref.png').write_bytes(PNG)
        run_id, path = self.create(graph={
            '1': {'class_type': 'Loader', 'inputs': {'model': 'model.safetensors'}},
            '3': {'class_type': 'LoadImage', 'inputs': {'image': 'ref.png'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}})
        with patch.object(r, 'cleanup_staging', return_value=[]):
            r.execute(run_id)
        manifest = read(path / 'manifest.json')
        staged = self.input / read(path / 'graph-api.json')['3']['inputs']['image']
        self.assertTrue(staged.exists())
        with patch.object(r, 'queue_rows', return_value=[{'input': str(staged.relative_to(self.input))}]):
            self.assertEqual(r.cleanup_staging(path, manifest, apply=True), [])
        self.assertTrue(staged.exists())
        self.assertEqual(len(r.cleanup_staging(path, manifest)), 1)
        self.assertTrue(staged.exists())
        staged.write_bytes(b'changed')
        self.assertEqual(r.cleanup_staging(path, manifest, apply=True), [])
        self.assertTrue(staged.exists())
        staged.write_bytes(PNG)
        r.cleanup_staging(path, manifest, apply=True)
        self.assertFalse(staged.exists())
        self.assertTrue(r.verify(path)['ok'])

    def test_offline_staging_cleanup_checks_backend_and_hash(self):
        (self.input / 'ref.png').write_bytes(PNG)
        run_id, path = self.create(graph={
            '1': {'class_type': 'Loader', 'inputs': {'model': 'model.safetensors'}},
            '3': {'class_type': 'LoadImage', 'inputs': {'image': 'ref.png'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}})
        with patch.object(r, 'cleanup_staging', return_value=[]):
            r.execute(run_id)
        manifest = read(path / 'manifest.json')
        staged = self.input / read(path / 'graph-api.json')['3']['inputs']['image']
        with patch.object(cleanup, 'idle_comfy_inputs', return_value={self.input.resolve()}):
            self.assertEqual(cleanup.offline_staging(path, manifest, True), [])
        self.assertTrue(staged.exists())
        with patch.object(cleanup, 'idle_comfy_inputs', return_value=set()):
            self.assertEqual(len(cleanup.offline_staging(path, manifest)), 1)
            staged.write_bytes(b'changed')
            self.assertEqual(cleanup.offline_staging(path, manifest, True), [])
            staged.write_bytes(PNG)
            self.assertEqual(len(cleanup.offline_staging(path, manifest, True)), 1)
        self.assertFalse(staged.exists())

    def test_cleanup_deduplicates_finished_inputs_without_touching_staged_runs(self):
        graph = {
            '1': {'class_type': 'Loader', 'inputs': {'model': 'model.safetensors'}},
            '3': {'class_type': 'LoadImage', 'inputs': {'image': '__UPLOAD__/0.png'}},
            '2': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'test'}}}
        upload = [{'binding': '3.image', 'name': '0.png', 'sha256': hashlib.sha256(PNG).hexdigest()}]
        run_id, path = self.create(graph=graph, uploads=upload)
        (path / 'inputs' / '0.png').write_bytes(PNG)
        with patch.object(r, 'cleanup_staging', return_value=[]):
            r.execute(run_id)
        untouched_id, untouched = self.create()
        (untouched / 'inputs' / 'keep.png').write_bytes(PNG)
        report = cleanup.clean()
        self.assertEqual(report['mode'], 'dry-run')
        self.assertTrue((untouched / 'inputs' / 'keep.png').exists())
        self.assertEqual(cleanup.clean(True)['counts']['state_succeeded'], 1)
        m = read(path / 'manifest.json')
        self.assertEqual((path / 'inputs' / '0.png').stat().st_ino, Path(m['inputs'][0]['path']).stat().st_ino)
        self.assertTrue(r.verify(path)['ok'])
        self.assertTrue((untouched / 'inputs' / 'keep.png').exists())
        with patch.object(r, 'launch', side_effect=lambda rid: r.public(read(r.run_path(rid) / 'manifest.json'))):
            again = r.rerun({'run_id': run_id})
        r.execute(again['run_id'])
        self.assertEqual(read(r.run_path(again['run_id']) / 'manifest.json')['state'], 'succeeded')
        self.assertTrue(r.verify(r.run_path(again['run_id']))['ok'])

    def test_output_escape_fails_archive(self):
        run_id, path = self.create()
        r.execute(run_id)
        history = self.history['prompt-1']
        history['outputs']['2']['images'][0]['filename'] = '/etc/passwd'
        m = read(path / 'manifest.json')
        with self.assertRaises(ValueError):
            r.collect(path, m, history)

    def test_changed_untracked_source_changes_stack_identity(self):
        a = copy.deepcopy(self.snapshot)
        b = copy.deepcopy(self.snapshot)
        (self.root / 'a').mkdir()
        (self.root / 'b').mkdir()
        r.capture_code(self.root / 'a', a)
        (self.code / 'main.py').write_text('print("changed")\n')
        r.capture_code(self.root / 'b', b)
        self.assertNotEqual(r.stack_identity(a), r.stack_identity(b))

    def test_repeated_source_snapshots_share_one_blob(self):
        for name in ('a', 'b', 'c'):
            (self.root / name).mkdir()
        first, second = copy.deepcopy(self.snapshot), copy.deepcopy(self.snapshot)
        r.capture_code(self.root / 'a', first)
        r.capture_code(self.root / 'b', second)
        archived = [self.root / name / 'source.tar.gz' for name in ('a', 'b', 'c')]
        self.assertEqual(first['source_archive_sha256'], second['source_archive_sha256'])
        self.assertEqual(archived[0].stat().st_ino, archived[1].stat().st_ino)
        self.assertEqual(len(list((self.archive / 'blobs').glob('*-source.tar.gz'))), 1)
        (self.code / 'main.py').write_text('print("changed")\n')
        r.capture_code(self.root / 'c', copy.deepcopy(self.snapshot))
        self.assertEqual(len(list((self.archive / 'blobs').glob('*-source.tar.gz'))), 2)
        self.assertNotEqual(archived[2].stat().st_ino, archived[0].stat().st_ino)

    def test_corrupt_source_blob_does_not_delete_new_archive(self):
        source = self.root / 'source.tar.gz'
        source.write_bytes(b'correct archive')
        blob = self.archive / 'blobs' / (digest(source) + '-source.tar.gz')
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b'corrupt archive')
        with self.assertRaisesRegex(ValueError, 'blob hash mismatch'):
            r.dedupe(source)
        self.assertEqual(source.read_bytes(), b'correct archive')

    def test_snapshot_includes_custom_node_ignored_by_parent_git(self):
        (self.code / '.gitignore').write_text('custom_nodes/\n')
        node = self.code / 'custom_nodes' / 'local_node'
        node.mkdir(parents=True)
        (node / '__init__.py').write_text('VALUE = 1\n')
        first, second = copy.deepcopy(self.snapshot), copy.deepcopy(self.snapshot)
        (self.root / 'a').mkdir()
        (self.root / 'b').mkdir()
        r.capture_code(self.root / 'a', first)
        (node / '__init__.py').write_text('VALUE = 2\n')
        r.capture_code(self.root / 'b', second)
        self.assertNotEqual(r.stack_identity(first), r.stack_identity(second))

    def test_tensorfold_library_changes_stack_identity(self):
        engine = self.root / 'libtf_h3.so'
        engine.write_bytes(b'first engine')
        proc = copy.deepcopy(self.proc)
        proc['environment']['TF_H3_LIB'] = str(engine)
        first, second = copy.deepcopy(self.snapshot), copy.deepcopy(self.snapshot)
        first['native_libraries'] = r.native_libraries(proc)
        engine.write_bytes(b'second engine')
        second['native_libraries'] = r.native_libraries(proc)
        first['sources'] = second['sources'] = []
        self.assertNotEqual(r.stack_identity(first), r.stack_identity(second))

    def test_tensorfold_build_metadata_must_match_engine(self):
        engine = self.root / 'libtf_h3.so'
        engine.write_bytes(b'engine')
        proc = copy.deepcopy(self.proc)
        proc['environment']['TF_H3_LIB'] = str(engine)
        metadata = {'engine_sha256': digest(engine), 'source_revision': 'abc', 'source_diff': 'patch'}
        write(self.root / 'build.json', metadata)
        self.assertEqual(r.native_libraries(proc)[0]['build'], metadata)
        engine.write_bytes(b'different engine')
        self.assertNotIn('build', r.native_libraries(proc)[0])

    def test_canonical_output_alias_and_recollection(self):
        run_id, path = self.create()
        r.execute(run_id)
        m = read(path / "manifest.json")
        output = m["outputs"][0]
        source, target = Path(output["source"]), Path(output["path"])
        self.assertTrue(source.is_symlink())
        self.assertEqual(source.resolve(), target)
        self.assertTrue(r.verify(path)["ok"])
        r.collect(path, m, self.history["prompt-1"])
        self.assertTrue(r.verify(path)["ok"])

    def test_alias_hash_mismatch_preserves_original(self):
        source, target = self.output / "original.png", self.output / "archive.png"
        source.write_bytes(PNG)
        target.write_bytes(b"wrong")
        with self.assertRaises(ValueError):
            r.output_alias(source, target, digest(source))
        self.assertFalse(source.is_symlink())
        self.assertEqual(source.read_bytes(), PNG)

    def test_frame_export_pruning_only_for_matching_video_source(self):
        graph = {
            '1': {'class_type': 'Decode', 'inputs': {}},
            '2': {'class_type': 'CreateVideo', 'inputs': {'images': ['1', 0]}},
            '3': {'class_type': 'SaveVideo', 'inputs': {'video': ['2', 0]}},
            '4': {'class_type': 'ImageFromBatch', 'inputs': {'image': ['1', 0], 'length': 124}},
            '5': {'class_type': 'SaveImage', 'inputs': {'images': ['4', 0]}},
            '6': {'class_type': 'SaveImage', 'inputs': {'images': ['1', 0]}},
        }
        self.assertEqual(library.prune_frame_saves(graph), ['5'])
        self.assertIn('6', graph)
        self.assertIn('3', graph)

    def test_library_links_canonical_output_without_copy(self):
        run_id, path = self.create()
        r.execute(run_id)
        manifest = read(path / 'manifest.json')
        output = manifest['outputs'][0]
        self.assertEqual(Path(output['path']).parent, path)
        self.assertFalse((path / 'outputs').exists())
        target = library.library_path(self.archive, manifest, output)
        self.assertTrue(target.is_file())
        self.assertEqual(target.name, run_id.split('-', 1)[0] + '-fixture.png')
        self.assertEqual(target.stat().st_ino, Path(output['path']).stat().st_ino)
        self.assertTrue(r.verify(path)['ok'])
        old = library.library_path(self.archive, manifest, output, legacy=True)
        import os
        os.link(Path(output['path']), old)
        self.assertTrue(library.index(self.archive, apply=True)['ok'])
        self.assertFalse(old.exists())
        self.assertTrue(target.is_file())

    def test_library_same_second_same_title_does_not_overwrite(self):
        first_id, first = self.create()
        r.execute(first_id)
        manifest = read(first / 'manifest.json')
        output = manifest['outputs'][0]
        source = Path(output['path'])
        other = first / 'second.png'
        other.write_bytes(b'different output')
        second = dict(manifest, run_id=first_id.split('-', 1)[0] + '-ffffffffffff')
        second_output = dict(output, path=str(other), sha256=digest(other))
        link = Path(library.link_output(self.archive, second, second_output))
        self.assertEqual(link.name, first_id.split('-', 1)[0] + '-fixture-2.png')
        self.assertEqual(link.read_bytes(), b'different output')
        self.assertEqual(Path(library.link_output(self.archive, second, second_output)), link)
        self.assertEqual(source.read_bytes(), PNG)

    def test_orphan_cleanup_requires_no_library_link_and_preserves_history_safety(self):
        run_id, path = self.create()
        r.execute(run_id)
        manifest = read(path / 'manifest.json')
        output = manifest['outputs'][0]
        source = Path(output['source'])
        library_link = library.library_path(self.archive, manifest, output)
        self.assertEqual(orphan_cleanup.clean_orphans()['counts']['has_library_entry'], 1)
        renamed = library_link.with_name('manually-renamed.png')
        library_link.rename(renamed)
        self.assertEqual(orphan_cleanup.clean_orphans()['counts']['has_library_entry'], 1)
        renamed.unlink()
        preview = orphan_cleanup.clean_orphans()
        self.assertEqual([x['run_id'] for x in preview['runs']], [run_id])
        self.assertTrue(path.exists())
        self.assertTrue(source.is_symlink())
        result = orphan_cleanup.clean_orphans(apply=True)
        self.assertEqual(result['counts']['deleted'], 1)
        self.assertFalse(path.exists())
        self.assertFalse(source.is_symlink())

    def test_orphan_cleanup_does_not_delete_an_unexpected_file_or_broken_library(self):
        run_id, path = self.create()
        r.execute(run_id)
        manifest = read(path / 'manifest.json')
        library.library_path(self.archive, manifest, manifest['outputs'][0]).unlink()
        (path / 'personal.txt').write_text('keep me')
        result = orphan_cleanup.clean_orphans(apply=True)
        self.assertEqual(len(result['errors']), 1)
        self.assertTrue((path / 'personal.txt').is_file())
        (self.archive / 'library').rename(self.archive / 'unavailable')
        with self.assertRaisesRegex(ValueError, 'library is missing'):
            orphan_cleanup.clean_orphans(apply=True)

    def test_orphan_cleanup_keeps_run_with_any_output_and_skips_locked_or_incomplete(self):
        import fcntl
        run_id, path = self.create()
        r.execute(run_id)
        m = read(path / 'manifest.json')
        first = m['outputs'][0]
        second_path = path / 'second.png'
        second_path.write_bytes(b'another image')
        second = dict(first, path=str(second_path), sha256=digest(second_path))
        m['outputs'].append(second)
        write(path / 'manifest.json', m)
        self.assertEqual(orphan_cleanup.clean_orphans()['counts']['has_library_entry'], 1)
        library.library_path(self.archive, m, first).unlink()
        with (path / '.lock').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(orphan_cleanup.clean_orphans(apply=True)['counts']['locked'], 1)
        m['state'] = 'uncertain'
        write(path / 'manifest.json', m)
        self.assertEqual(orphan_cleanup.clean_orphans(apply=True)['counts']['not_succeeded'], 1)
        self.assertTrue(path.exists())

    def test_rpc_preserves_recorded_errors_and_lists(self):
        from types import SimpleNamespace
        for value in [{'state': 'failed', 'error': 'Renderer rejected graph'}, {'state': 'succeeded', 'error': None}, []]:
            with patch.object(render.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(value))):
                self.assertEqual(render.rpc(SimpleNamespace(host='spark-2'), 'status'), value)
        with patch.object(render.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps({'__rpc_error__': 'bad request'}))):
            with self.assertRaisesRegex(RuntimeError, 'bad request'):
                render.rpc(SimpleNamespace(host='spark-2'), 'status')

    def test_remote_path_quotes_shell_metacharacters_and_resolves_home(self):
        import shlex
        self.assertEqual(render.remote_path('~/my recorder/recorder.py'), '"$HOME"/' + shlex.quote('my recorder/recorder.py'))
        self.assertEqual(render.remote_path('/opt/my recorder/recorder.py'), shlex.quote('/opt/my recorder/recorder.py'))
        self.assertEqual(render.remote_path('~/$(touch bad)'), '"$HOME"/' + shlex.quote('$(touch bad)'))
        with self.assertRaises(ValueError):
            render.remote_path('relative/recorder.py')


if __name__ == '__main__':
    unittest.main()
