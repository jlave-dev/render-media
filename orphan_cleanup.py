"""Preview or remove complete runs whose outputs no longer appear in the library."""
import fcntl
import os
import shutil
from collections import Counter
from pathlib import Path

import recorder as r
from common import digest, read


def library_inodes(root):
    library = root / 'library'
    if library.is_symlink() or not library.is_dir():
        raise ValueError('Media library is missing or is a symlink; refusing cleanup')
    inodes = set()
    def failed(error):
        raise error
    for folder, dirs, files in os.walk(library, onerror=failed, followlinks=False):
        for name in dirs:
            if (Path(folder) / name).is_symlink():
                raise ValueError('Library contains a symlinked directory; refusing cleanup')
        for name in files:
            st = (Path(folder) / name).stat()
            inodes.add((st.st_dev, st.st_ino))
    return inodes


def checked_run(path, manifest, inodes):
    if path.is_symlink() or manifest.get('run_id') != path.name or manifest.get('state') != 'succeeded':
        raise ValueError('Not a completed run with a matching run ID')
    outputs = manifest.get('outputs', [])
    if not outputs:
        raise ValueError('No recorded outputs')
    allowed = {'.lock', 'manifest.json', 'graph-original.json', 'graph-api.json', 'workflow-ui.json',
               'node-schemas.json', 'runtime.json', 'history.json', 'recorder.log', 'source.tar.gz',
               'memory-samples.csv', 'rejection.json'}
    allowed_inputs = {Path(i['path']).name for i in manifest.get('inputs') or []}
    allowed_inputs.update(i['name'] for i in manifest.get('uploads') or [])
    allowed_outputs = set()
    for output in outputs:
        file = Path(output['path'])
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(path.resolve()):
            raise ValueError('Recorded output missing or outside run: ' + str(file))
        st = file.stat()
        if (st.st_dev, st.st_ino) in inodes:
            return None
        if digest(file) != output['sha256']:
            raise ValueError('Recorded output hash mismatch: ' + str(file))
        relative = file.relative_to(path)
        if len(relative.parts) == 1:
            allowed.add(relative.name)
        elif len(relative.parts) == 2 and relative.parts[0] == 'outputs':
            allowed_outputs.add(relative.name)
        else:
            raise ValueError('Unexpected output location: ' + str(file))
    for entry in path.iterdir():
        if entry.name in {'inputs', 'outputs'} and entry.is_dir() and not entry.is_symlink():
            expected = allowed_inputs if entry.name == 'inputs' else allowed_outputs
            for child in entry.iterdir():
                if child.is_symlink() or not child.is_file() or child.name not in expected:
                    raise ValueError('Unexpected file in run: ' + str(child))
        elif entry.name not in allowed or entry.is_symlink() or not entry.is_file():
            raise ValueError('Unexpected file in run: ' + str(entry))
    for record in manifest.get('inputs') or []:
        file = Path(record['path'])
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to((path / 'inputs').resolve()):
            raise ValueError('Recorded input missing or outside run: ' + str(file))
        if digest(file) != record['sha256']:
            raise ValueError('Recorded input hash mismatch: ' + str(file))
    return outputs


def clean_orphans(apply=False):
    inodes = library_inodes(r.ROOT)
    counts, candidates, errors = Counter(), [], []
    for mf in sorted((r.ROOT / 'runs').glob('*/*/*/*/manifest.json')):
        run = mf.parent
        try:
            lock_path = run / '.lock'
            if lock_path.is_symlink() or not lock_path.is_file():
                counts['missing_lock'] += 1
                continue
            with lock_path.open('r+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    counts['locked'] += 1
                    continue
                manifest = read(mf)
                if manifest.get('state') != 'succeeded':
                    counts['not_succeeded'] += 1
                    continue
                outputs = checked_run(run, manifest, inodes)
                if outputs is None:
                    counts['has_library_entry'] += 1
                    continue
                candidate = {'run_id': manifest['run_id'], 'title': manifest.get('title'),
                             'run_path': str(run), 'outputs': [o['path'] for o in outputs],
                             'output_bytes': sum(o.get('bytes', 0) for o in outputs)}
                candidates.append(candidate)
                if apply:
                    # Recheck against a fresh scan immediately before deletion.
                    if checked_run(run, manifest, library_inodes(r.ROOT)) is None:
                        counts['restored_to_library'] += 1
                        candidates.pop()
                        continue
                    aliases = []
                    runtime_file = run / 'runtime.json'
                    if runtime_file.is_file():
                        output_root = Path(read(runtime_file)['process']['output']).resolve() / 'media-runs' / manifest['run_id']
                        for output in outputs:
                            source = Path(output.get('source', ''))
                            if source.is_symlink() and source.parent.resolve().is_relative_to(output_root.resolve()) and source.resolve() == Path(output['path']).resolve():
                                aliases.append(source)
                    shutil.rmtree(run)
                    for alias in aliases:
                        if alias.is_symlink():
                            alias.unlink()
                    counts['deleted'] += 1
                else:
                    counts['candidates'] += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'run_id': run.name, 'error': str(exc)})
    return {'ok': not errors, 'mode': 'apply' if apply else 'dry-run', 'counts': dict(counts),
            'runs': candidates, 'errors': errors}
