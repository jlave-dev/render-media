#!/usr/bin/env python3
"""Browseable hard links and conservative detection of video frame exports."""
import fcntl
import os
import re
from collections import Counter
from pathlib import Path
from common import digest, read


def frame_save_nodes(graph):
    video_inputs = set()
    for node in graph.values():
        if node.get('class_type') != 'SaveVideo':
            continue
        link = node.get('inputs', {}).get('video')
        if not isinstance(link, list) or len(link) != 2:
            continue
        video = graph.get(str(link[0]), {})
        images = video.get('inputs', {}).get('images')
        if video.get('class_type') == 'CreateVideo' and isinstance(images, list) and len(images) == 2:
            video_inputs.add(tuple(images))
    result = set()
    for node_id, node in graph.items():
        if node.get('class_type') != 'SaveImage':
            continue
        link = node.get('inputs', {}).get('images')
        if not isinstance(link, list) or len(link) != 2:
            continue
        batch = graph.get(str(link[0]), {})
        inputs = batch.get('inputs', {})
        source = inputs.get('image')
        if (batch.get('class_type') == 'ImageFromBatch' and isinstance(source, list)
                and len(source) == 2 and tuple(source) in video_inputs
                and isinstance(inputs.get('length'), int) and inputs['length'] > 1):
            result.add(node_id)
    return result


def prune_frame_saves(graph):
    removable = frame_save_nodes(graph)
    # Preserve save nodes referenced by another node or supplied as an explicit input.
    for node_id, node in graph.items():
        if node_id in removable:
            continue
        for value in node.get('inputs', {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in removable:
                removable.discard(str(value[0]))
    for node_id in removable:
        del graph[node_id]
    return sorted(removable)


def frame_outputs(path, manifest):
    graph_file = path / 'graph-api.json'
    if not graph_file.is_file():
        return []
    ids = frame_save_nodes(read(graph_file))
    return [o for o in manifest.get('outputs', []) if any(
        re.match(r'^\d+-node-' + re.escape(node) + r'_', Path(o['path']).name) for node in ids)]


def library_path(root, manifest, output, legacy=False):
    run_id = manifest['run_id']
    title = re.sub(r'[^a-z0-9]+', '-', manifest.get('title', '').lower()).strip('-')[:60] or 'render'
    name = Path(output['path']).name
    directory = root / 'library' / run_id[:4] / run_id[4:6] / run_id[6:8]
    if legacy:
        return directory / (title + '-' + run_id + '-' + name)
    return directory / (run_id.split('-', 1)[0] + '-' + title + Path(name).suffix.lower())


def link_output(root, manifest, output):
    source = Path(output['path'])
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to((root / 'runs').resolve()):
        raise ValueError('Output is missing or outside the archive: ' + str(source))
    if digest(source) != output['sha256']:
        raise ValueError('Output hash mismatch: ' + str(source))
    base = library_path(root, manifest, output)
    base.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pattern = re.compile(re.escape(base.stem) + r'(?:-\d+)?' + re.escape(base.suffix) + r'$')
    existing = [p for p in base.parent.glob(base.stem + '*' + base.suffix)
                if pattern.fullmatch(p.name) and not p.is_symlink() and p.is_file() and os.path.samefile(p, source)]
    if existing:
        target = existing[0]
    else:
        number = 1
        while True:
            target = base if number == 1 else base.with_name(base.stem + '-' + str(number) + base.suffix)
            try:
                os.link(source, target)
                break
            except FileExistsError:
                if not target.is_symlink() and target.is_file() and os.path.samefile(source, target):
                    break
                number += 1
    run_id = manifest['run_id']
    title = re.sub(r'[^a-z0-9]+', '-', manifest.get('title', '').lower()).strip('-')[:60] or 'render'
    old_paths = [library_path(root, manifest, output, legacy=True),
                 base.parent / (run_id + '-' + title + '-' + source.name)]
    for old in old_paths:
        if old.is_symlink():
            raise ValueError('Old library path is a symlink: ' + str(old))
        if old.exists():
            if not old.is_file() or not os.path.samefile(source, old):
                raise ValueError('Old library path is not a link to the recorded output: ' + str(old))
            old.unlink()
    return str(target)


def index(root, apply=False):
    counts = Counter()
    errors, previews = [], []
    for manifest_path in sorted((root / 'runs').glob('*/*/*/*/manifest.json')):
        path = manifest_path.parent
        try:
            lock_path = path / '.lock'
            if not lock_path.is_file():
                counts['missing_locks'] += 1
                continue
            with lock_path.open('r+') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    counts['locked_runs'] += 1
                    continue
                manifest = read(manifest_path)
                if manifest.get('state') != 'succeeded':
                    counts['other_runs_skipped'] += 1
                    continue
                outputs = manifest.get('outputs', [])
                frames = frame_outputs(path, manifest)
                videos = [o for o in outputs if Path(o['path']).suffix.lower() in {'.mp4', '.webm', '.mov'}]
                omitted = {o['path'] for o in frames} if videos else set()
                if omitted:
                    previews.append({'run_id': manifest['run_id'], 'title': manifest.get('title'),
                                     'frames': len(frames), 'bytes': sum(o.get('bytes', 0) for o in frames),
                                     'frame_paths': [o['path'] for o in frames],
                                     'video_paths': [o['path'] for o in videos]})
                    counts['frame_files'] += len(frames)
                    counts['frame_bytes'] += sum(o.get('bytes', 0) for o in frames)
                for output in outputs:
                    if output['path'] in omitted:
                        continue
                    if apply:
                        link_output(root, manifest, output)
                    counts['library_files'] += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append({'run_id': path.name, 'error': str(exc)})
    return {'ok': not errors, 'mode': 'apply' if apply else 'dry-run', 'counts': dict(counts),
            'frame_review': previews, 'errors': errors}
