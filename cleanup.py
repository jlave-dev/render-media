#!/usr/bin/env python3
"""Deduplicate finished runs and remove verified input staging copies."""
import argparse
import fcntl
import json
from collections import Counter
from pathlib import Path

import recorder as r
from common import contained, digest, read


def idle_comfy_inputs():
    """Return live ComfyUI input roots only when all known queues are idle."""
    roots = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not any(Path(arg).name == 'main.py' for arg in argv[:3]):
                continue
            cwd = (proc / 'cwd').resolve()
            if not (cwd / 'folder_paths.py').is_file() or not (cwd / 'nodes.py').is_file():
                continue
            port = int(argv[argv.index('--port') + 1]) if '--port' in argv else 8188
            backend = next((name for name, value in r.PORTS.items() if value == port), None)
            if backend is None:
                raise ValueError('A ComfyUI process uses an unknown port')
            live = r.process(backend)
            if live['pid'] != int(proc.name) or r.queue_rows(backend):
                raise ValueError('A ComfyUI queue is active or ambiguous')
            roots.add(Path(live['input']).resolve())
        except PermissionError as exc:
            raise ValueError('Cannot inspect a possible ComfyUI process') from exc
        except (FileNotFoundError, ProcessLookupError, UnicodeError, IndexError):
            continue
    return roots


def offline_staging(path, manifest, apply=False):
    """Clean a retired backend's inputs only while all ComfyUI queues are idle."""
    proc = read(path / 'runtime.json')['process']
    root = Path(proc['input']).resolve()
    if root in idle_comfy_inputs():
        return []
    graph = read(path / 'graph-api.json')
    result = []
    for record in manifest.get('inputs', []):
        node, field = record['binding'].split('.', 1)
        name = graph[node]['inputs'][field]
        if not isinstance(name, str) or not name.startswith('media-runs/' + manifest['run_id'] + '/'):
            continue
        raw = root / name
        if raw.is_symlink():
            continue
        staged = contained(root, name)
        archived = Path(record['path'])
        if not staged.is_file() or not archived.is_file():
            continue
        if digest(staged) != record['sha256'] or digest(archived) != record['sha256']:
            continue
        size = staged.stat().st_size
        if apply:
            if root in idle_comfy_inputs():
                continue
            staged.unlink()
        result.append({'path': str(staged), 'bytes': size})
    return result


def clean(apply=False):
    counts = Counter()
    old_inodes = set()
    migrated = set()
    errors = []
    review = []
    input_inodes = {}
    for manifest_file in sorted((r.ROOT / 'runs').glob('*/*/*/*/manifest.json')):
        path = manifest_file.parent
        try:
            lock_path = path / '.lock'
            if not lock_path.exists():
                counts['missing_run_lock'] += 1
                continue
            with lock_path.open('r+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    counts['locked_runs'] += 1
                    continue
                m = read(manifest_file)
                counts['state_' + m['state']] += 1
                if m['state'] != 'succeeded':
                    review.append({'run_id': m['run_id'], 'state': m['state'],
                                   'created_at': m.get('created_at'), 'title': m.get('title')})
                    continue
                records = [(x['path'], x['sha256']) for x in m.get('inputs', [])]
                records += [(str(contained(path / 'inputs', x['name'])), x['sha256'])
                            for x in m.get('uploads', [])]
                checked = []
                for name, expected in records:
                    f = Path(name)
                    if f.is_symlink() or not f.is_file() or not f.resolve().is_relative_to((path / 'inputs').resolve()):
                        raise ValueError('Input missing or outside run: ' + name)
                    if digest(f) != expected:
                        raise ValueError('Input hash mismatch: ' + name)
                    checked.append((f, expected))
                for f, expected in checked:
                    st = f.stat()
                    counts['input_paths'] += 1
                    counts['input_path_bytes'] += st.st_size
                    old_inodes.add((st.st_dev, st.st_ino))
                    migrated.add(expected)
                    input_inodes.setdefault(expected, {})[(st.st_dev, st.st_ino)] = st.st_size
                    if apply:
                        r.dedupe_input(f)
                try:
                    try:
                        candidates = r.cleanup_staging(path, m, apply=apply)
                    except ValueError:
                        candidates = offline_staging(path, m, apply=apply)
                    if not candidates:
                        candidates = offline_staging(path, m, apply=apply)
                    counts['staging_files'] += len(candidates)
                    counts['staging_bytes'] += sum(x['bytes'] for x in candidates)
                except Exception as exc:
                    counts['staging_runs_skipped'] += 1
                    errors.append({'run': m['run_id'], 'error': 'staging: ' + str(exc)})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            counts['runs_skipped'] += 1
            errors.append({'run': path.name, 'error': str(exc)})
    counts['potential_duplicate_input_bytes'] = sum(
        sum(inodes.values()) - next(iter(inodes.values())) for inodes in input_inodes.values()
    )
    if apply:
        # Legacy blobs may be dropped only after their last run link is replaced.
        for blob in (r.ROOT / 'blobs').glob('*'):
            if blob.is_symlink() or not blob.is_file():
                continue
            st = blob.stat()
            if st.st_nlink != 1 or (st.st_dev, st.st_ino) not in old_inodes:
                continue
            sha = blob.name.split('-', 1)[0]
            replacement = r.ROOT / 'blobs' / 'inputs' / sha
            if sha in migrated and replacement.is_file() and digest(blob) == sha and digest(replacement) == sha:
                blob.unlink()
                counts['legacy_blobs_removed'] += 1
                counts['legacy_blob_bytes'] += st.st_size
    return {'ok': not counts['runs_skipped'], 'mode': 'apply' if apply else 'dry-run',
            'counts': dict(counts), 'review': review, 'errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = clean(args.apply)
    print(json.dumps(result, indent=2))
    if result['counts'].get('runs_skipped'):
        raise SystemExit(1)
