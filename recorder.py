#!/usr/bin/env python3
"""Remote recorder. No service lifecycle actions or GPU imports."""
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from common import contained, digest, extension, graph_value, leaves, now, read, slug, token, write
from memory_trace import MemoryTrace

ROOT = Path.home() / 'media'
HERE = Path(__file__).resolve().parent
PORTS = {'comfy': 8188, 'image': 8196, 'dual': 8195}
MODEL_EXTENSIONS = {'.safetensors', '.gguf', '.ckpt', '.pt', '.pth', '.bin', '.sft', '.onnx'}


def api(backend, path, payload=None):
    req = urllib.request.Request('http://127.0.0.1:' + str(PORTS[backend]) + path,
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def run_path(run_id):
    if not re.fullmatch(r'\d{8}T\d{6}Z-[a-f0-9]{12}', run_id):
        raise ValueError('Invalid run ID')
    path = ROOT / 'runs' / run_id[:4] / run_id[4:6] / run_id[6:8] / run_id
    if not (path / 'manifest.json').is_file():
        raise ValueError('Run not found: ' + run_id)
    return path


def save(path, manifest, state=None, **updates):
    manifest.update(updates, updated_at=now())
    if state:
        manifest['state'] = state
    write(path / 'manifest.json', manifest)
    return manifest


def public(manifest):
    return {k: manifest[k] for k in ['run_id', 'project', 'title', 'state', 'path', 'backend', 'created_at', 'prompt_id', 'outputs', 'error', 'gaps'] if k in manifest}


def create(request):
    slug(request['project'])
    if not isinstance(request['title'], str) or not 1 <= len(request['title']) <= 300:
        raise ValueError('Title must contain 1-300 characters')
    if request['backend'] not in PORTS:
        raise ValueError('Unknown backend')
    graph = request['graph']
    if not graph or not isinstance(graph, dict) or any(not isinstance(n, dict) or not isinstance(n.get('class_type'), str) or not isinstance(n.get('inputs'), dict) for n in graph.values()):
        raise ValueError('Expected an API graph')
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:12]
    path = ROOT / 'runs' / run_id[:4] / run_id[4:6] / run_id[6:8] / run_id
    (path / 'inputs').mkdir(parents=True, mode=0o700)
    manifest = {k: request.get(k) for k in ['project', 'title', 'recipe', 'backend', 'requester', 'uploads', 'allow_busy', 'allow_incomplete', 'keep_frames']}
    manifest.update(schema_version=1, run_id=run_id, path=str(path), created_at=now(), state='staged',
                    graph_sha256=None, inputs=[], assets=[], outputs=[], gaps=[],
                    recorder_sha256=digest(HERE / 'recorder.py'))
    write(path / 'graph-original.json', graph)
    if request.get('workflow') is not None:
        write(path / 'workflow-ui.json', request['workflow'])
    save(path, manifest)
    return public(manifest)


def process(backend):
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not any(Path(a).name == 'main.py' for a in argv[:3]):
                continue
            port = int(argv[argv.index('--port') + 1]) if '--port' in argv else 8188
            if port != PORTS[backend]:
                continue
            cwd = str((proc / 'cwd').resolve())
            index = next(i for i, a in enumerate(argv) if Path(a).name == 'main.py')
            flags = argv[index + 1:]
            def directory(flag, default):
                return str(Path(argv[argv.index(flag) + 1]).resolve()) if flag in argv else str(Path(cwd) / default)
            env = dict(item.split('=', 1) for item in (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
            matches.append({'pid': int(proc.name), 'argv': argv, 'cwd': cwd, 'flags': flags,
                            'python_executable': argv[0], 'input': directory('--input-directory', 'input'),
                            'output': directory('--output-directory', 'output'), 'temp': directory('--temp-directory', 'temp'),
                            'environment': {k: v for k, v in env.items() if k.startswith(('CUDA_', 'TORCH_', 'PYTORCH_', 'CUBLAS_', 'NVIDIA_', 'COMFY_', 'TF_H3_'))}})
        except (OSError, ValueError, UnicodeError):
            continue
    if len(matches) != 1:
        raise ValueError('Expected one live ComfyUI process on port ' + str(PORTS[backend]) + '; found ' + str(len(matches)) + '. No services were started.')
    return matches[0]


def runtime(backend):
    proc = process(backend)
    result = subprocess.run([proc['python_executable'], str(HERE / 'inspect_runtime.py')],
                            input=json.dumps(proc), capture_output=True, text=True, check=True, timeout=60)
    snapshot = json.loads(result.stdout)
    snapshot.update(process=proc, native_libraries=native_libraries(proc), system_stats=api(backend, '/system_stats'), hostname=os.uname().nodename,
                    driver=subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,uuid', '--format=csv,noheader'],
                                          capture_output=True, text=True, timeout=10).stdout.strip())
    return snapshot


def native_libraries(proc):
    configured = proc['environment'].get('TF_H3_LIB')
    path = Path(configured) if configured else Path(proc['cwd']) / 'custom_nodes/tensorfold_h3/libtf_h3.so'
    if not path.is_absolute():
        path = Path(proc['cwd']) / path
    if not configured and not path.is_file():
        return []
    path = path.resolve(strict=True)
    before = signature(path)
    sha = digest(path)
    if signature(path) != before:
        raise ValueError('Native library changed while hashing: ' + str(path))
    result = {'path': str(path), 'sha256': sha, 'bytes': before[2]}
    metadata = path.parent / 'build.json'
    if metadata.is_file():
        build = read(metadata)
        if build.get('engine_sha256') == sha:
            result['build'] = build
    return [result]


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def dedupe(file):
    """Store one copy of a repeated artifact under ROOT/blobs and leave a hardlink at file."""
    file = Path(file)
    sha = digest(file)
    store = ROOT / 'blobs'
    store.mkdir(parents=True, mode=0o700, exist_ok=True)
    target = store / (sha + '-' + file.name)
    try:
        os.link(file, target)
    except FileExistsError:
        if digest(target) != sha:
            raise ValueError('Source archive blob hash mismatch: ' + str(target))
    if not os.path.samefile(file, target):
        temporary = file.with_name('.link-' + uuid.uuid4().hex)
        try:
            os.link(target, temporary)
            os.replace(temporary, file)
        finally:
            temporary.unlink(missing_ok=True)
    return sha


def dedupe_input(file):
    """Atomically share archived input bytes by content, regardless of filename."""
    file = Path(file)
    sha = digest(file)
    store = ROOT / 'blobs' / 'inputs'
    store.mkdir(parents=True, mode=0o700, exist_ok=True)
    blob = store / sha
    try:
        os.link(file, blob)
    except FileExistsError:
        if digest(blob) != sha:
            raise ValueError('Input blob hash mismatch: ' + str(blob))
    if os.path.samefile(file, blob):
        return sha
    temporary = file.with_name('.link-' + uuid.uuid4().hex)
    try:
        os.link(blob, temporary)
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)
    return sha


def asset_record(path):
    path = Path(path).resolve(strict=True)
    before = signature(path)
    key = hashlib.sha256(str(path).encode()).hexdigest()
    cache = ROOT / '.hash-cache' / (key + '.json')
    cached = read(cache) if cache.exists() else None
    if not cached or cached['signature'] != before:
        sha = digest(path)
        if signature(path) != before:
            raise ValueError('File changed while hashing: ' + str(path))
        cached = {'signature': before, 'sha256': sha}
        write(cache, cached)
    return {'path': str(path), 'sha256': cached['sha256'], 'bytes': before[2], 'signature': before}


def model_index(snapshot):
    index = {}
    for category, roots in snapshot['model_paths'].items():
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            for directory, dirs, files in os.walk(root, followlinks=True):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for filename in files:
                    p = Path(directory) / filename
                    relative = str(p.relative_to(root))
                    for key in {relative, filename}:
                        index.setdefault(key, set()).add((category, str(p.resolve())))
    return index


def capture_code(path, snapshot):
    """Preserve working source without changing the live checkouts."""
    cwd = Path(snapshot['process']['cwd'])
    roots = [('comfyui', cwd)]
    flags = snapshot['process']['flags']
    disabled = '--disable-all-custom-nodes' in flags
    allowed = []
    if '--whitelist-custom-nodes' in flags:
        for value in flags[flags.index('--whitelist-custom-nodes') + 1:]:
            if value.startswith('--'):
                break
            allowed.append(value)
    custom = cwd / 'custom_nodes'
    for item in custom.iterdir() if custom.exists() else []:
        if not item.name.startswith(('.', '__')) and (not disabled or item.name in allowed):
            roots.append(('custom/' + item.name, item.resolve()))
    sources = []
    archive = path / 'source.tar.gz'
    # mtime=0 makes the archive byte-identical for an unchanged stack, so repeated renders share one blob.
    raw = open(archive, 'wb')
    gz = gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0)
    with tarfile.open(fileobj=gz, mode='w') as tar:
        for label, root in roots:
            top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
            own_repo = top.returncode == 0 and Path(top.stdout.strip()).resolve() == root
            revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True) if own_repo else None
            diff = subprocess.run(['git', '-C', str(root), 'diff', '--binary', 'HEAD'], capture_output=True) if own_repo else None
            source = {'label': label, 'path': str(root), 'commit': (revision.stdout.strip() or None) if revision else None,
                      'diff_sha256': hashlib.sha256(diff.stdout).hexdigest() if diff else None}
            sources.append(source)
            hashes = {}
            # Git paths include untracked source and preserve local changes. Runtime binaries are
            # identified by package versions, not copied into every generation archive.
            listing = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], capture_output=True) if own_repo else None
            files = [root / os.fsdecode(p) for p in listing.stdout.split(b'\0') if p] if listing and listing.returncode == 0 else list(root.rglob('*')) if root.is_dir() else [root]
            for file in files:
                if not file.is_file() or file.is_symlink() or any(part in {'.git', 'models', 'node_modules', '__pycache__', 'input', 'output', 'temp', 'user', '.venv', 'venv'} for part in file.relative_to(root).parts):
                    continue
                if file.name.startswith('.env') or re.search(r'credential|api.key|private.key|auth\.json', file.name, re.I):
                    continue
                if file.suffix.lower() in MODEL_EXTENSIONS or file.stat().st_size > 16 * 1024 * 1024:
                    continue
                hashes[str(file.relative_to(root))] = digest(file)
                tar.add(file, arcname=label + '/' + str(file.relative_to(root)), recursive=False)
            source['files_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    gz.close()
    raw.close()
    snapshot['sources'] = sources
    snapshot['source_archive_sha256'] = dedupe(archive)
    return sources


def input_path(value, proc):
    match = re.fullmatch(r'(.*?)\s*\[(input|output|temp)\]', value)
    root = proc[match[2]] if match else proc['input']
    name = match[1].rstrip() if match else value
    return contained(root, name)


def prepare(path, manifest):
    (path / 'inputs').mkdir(mode=0o700, exist_ok=True)
    snapshot = runtime(manifest['backend'])
    schemas = api(manifest['backend'], '/object_info')
    graph = read(path / 'graph-original.json')
    from library import prune_frame_saves
    removed_frame_nodes = [] if manifest.get('keep_frames') else prune_frame_saves(graph)
    gaps, assets, inputs = [], [], []
    index = model_index(snapshot)
    proc = snapshot['process']
    upload_map = {u['binding']: u for u in manifest.get('uploads') or []}
    saved_outputs = 0
    for node_id, node in graph.items():
        kind = node['class_type']
        if kind not in schemas:
            raise ValueError('Backend has no node class: ' + kind)
        schema = schemas[kind]
        fields = {**schema.get('input', {}).get('required', {}), **schema.get('input', {}).get('optional', {})}
        for field, value in list(node['inputs'].items()):
            if not isinstance(value, str):
                continue
            binding = node_id + '.' + field
            spec = fields.get(field, [])
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            upload = upload_map.get(binding)
            if upload or options.get('image_upload') or options.get('video_upload') or options.get('audio_upload') or (kind.startswith('Load') and field in {'image', 'video', 'audio', 'mask'}):
                if upload:
                    source = contained(path / 'inputs', upload['name'])
                    if digest(source) != upload['sha256']:
                        raise ValueError('Upload hash mismatch: ' + binding)
                    dedupe_input(source)
                else:
                    source = input_path(value, proc)
                name = 'node-' + token(node_id) + '-' + token(field) + extension(source)
                target = path / 'inputs' / name
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
                    dedupe_input(target)
                record = asset_record(target)
                record.update(binding=binding, original_value=value, original_path=str(source))
                inputs.append(record)
                backend_name = 'media-runs/' + manifest['run_id'] + '/' + name
                destination = contained(proc['input'], backend_name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, destination)
                node['inputs'][field] = backend_name
                continue
            if field == 'filename_prefix':
                node['inputs'][field] = 'media-runs/' + manifest['run_id'] + '/node-' + token(node_id)
                saved_outputs += 1
                continue
            matches = index.get(value, set())
            if matches:
                paths = {p for _, p in matches}
                if len(paths) != 1:
                    gaps.append('Ambiguous asset ' + binding + ': ' + value)
                else:
                    record = asset_record(next(iter(paths)))
                    record.update(binding=binding, value=value, categories=sorted({c for c, _ in matches}))
                    assets.append(record)
            elif Path(value).suffix.lower() in MODEL_EXTENSIONS:
                gaps.append('Unresolved asset ' + binding + ': ' + value)
            elif re.search(r'(^|_)(file|filename|path|directory|folder|url)($|_)', field, re.I):
                gaps.append('Uncaptured filesystem/external input ' + binding + ': ' + value)
        for field, value in leaves(node['inputs']):
            if '.' in field and Path(value).suffix.lower() in MODEL_EXTENSIONS:
                gaps.append('Nested asset needs explicit capture: ' + node_id + '.' + field)
        seed = node['inputs'].get('noise_seed')
        fixed_noise = kind == 'RandomNoise' and type(seed) is int and 0 <= seed <= 0xffffffffffffffff
        if not fixed_noise and re.search(r'wildcard|random|ollama|llm|script|http|url|raylight', kind, re.I):
            gaps.append('Dynamic/external node needs resolved-value capture: ' + kind)
    if not saved_outputs:
        gaps.append('No filename_prefix save node; outputs depend on node-specific history reporting')
    if manifest['backend'] == 'dual':
        gaps.append('Distributed worker environment is not captured by this single-host recorder')
    capture_code(path, snapshot)
    write(path / 'runtime.json', snapshot)
    write(path / 'node-schemas.json', {n['class_type']: schemas[n['class_type']] for n in graph.values()})
    write(path / 'graph-api.json', graph)
    if not inputs and not manifest.get('uploads'):
        (path / 'inputs').rmdir()
    save(path, manifest, 'prepared', graph_sha256=digest(path / 'graph-api.json'), assets=assets,
         omitted_frame_save_nodes=removed_frame_nodes,
         inputs=inputs, gaps=gaps, runtime_sha256=digest(path / 'runtime.json'),
         provenance='incomplete' if gaps else 'static-graph-captured',
         limitations=['Seeds and a matching stack do not guarantee pixel-identical GPU results.',
                      'Weights remain in the shared library; back them up before deleting or replacing them.',
                      'Source snapshots reflect disk at submission; an existing process may retain previously imported code.',
                      'Package versions are captured; installed binary wheels and external services are not archived.'])
    if gaps and not manifest.get('allow_incomplete'):
        raise ValueError('Provenance incomplete; inspect gaps with show. Explicit --allow-incomplete is required to submit.')
    return manifest


def queue_rows(backend):
    q = api(backend, '/queue')
    return q.get('queue_running', []) + q.get('queue_pending', [])


def locate(manifest):
    history = api(manifest['backend'], '/history/' + manifest['prompt_id']) if manifest.get('prompt_id') else api(manifest['backend'], '/history?max_items=10000')
    for prompt_id, entry in history.items():
        prompt = entry.get('prompt', [])
        extra = prompt[3] if len(prompt) > 3 and isinstance(prompt[3], dict) else {}
        if prompt_id == manifest.get('prompt_id') or extra.get('media_render_run_id') == manifest['run_id']:
            return prompt_id, entry
    for row in queue_rows(manifest['backend']):
        if row[1] == manifest.get('prompt_id') or (len(row) > 3 and isinstance(row[3], dict) and row[3].get('media_render_run_id') == manifest['run_id']):
            return row[1], None
    return None, None


def cleanup_staging(path, manifest, apply=False):
    """Remove only verified run-owned staging inputs that are not queued."""
    if manifest.get('state') != 'succeeded':
        return []
    proc = read(path / 'runtime.json')['process']
    live = process(manifest['backend'])
    if Path(live['input']).resolve() != Path(proc['input']).resolve():
        return []
    queued = json.dumps(queue_rows(manifest['backend']))
    graph = read(path / 'graph-api.json')
    results = []
    for record in manifest.get('inputs', []):
        node, field = record['binding'].split('.', 1)
        value = graph[node]['inputs'][field]
        prefix = 'media-runs/' + manifest['run_id'] + '/'
        if not isinstance(value, str) or not value.startswith(prefix) or value in queued:
            continue
        raw_candidate = Path(proc['input']) / value
        if raw_candidate.is_symlink():
            continue
        candidate = contained(proc['input'], value)
        archived = Path(record['path'])
        if not candidate.is_file() or not archived.is_file():
            continue
        if digest(candidate) != record['sha256'] or digest(archived) != record['sha256']:
            continue
        size = candidate.stat().st_size
        if apply:
            candidate.unlink()
        results.append({'path': str(candidate), 'bytes': size})
    return results


def output_alias(source, target, expected):
    """Preserve renderer history as an alias to the canonical archive."""
    if digest(source) != expected or digest(target) != expected:
        raise ValueError("Output hash mismatch; original retained")
    if source.is_symlink() and source.resolve() == target.resolve():
        return
    alias = source.with_name(source.name + ".archive-link")
    try:
        alias.symlink_to(target.resolve())
        os.replace(alias, source)
    finally:
        alias.unlink(missing_ok=True)


def collect(path, manifest, history):
    write(path / 'history.json', history)
    outputs = []
    seen = set()
    previous = {item['source']: Path(item['path']) for item in manifest.get('outputs', [])
                if Path(item['path']).resolve().is_relative_to(path.resolve())}
    def files(value):
        if isinstance(value, dict):
            if isinstance(value.get('filename'), str):
                yield value
            else:
                for child in value.values():
                    yield from files(child)
        elif isinstance(value, list):
            for child in value:
                yield from files(child)
    proc = read(path / 'runtime.json')['process']
    for descriptor in files(history.get('outputs', {})):
        file_type = descriptor.get('type', 'output')
        if file_type not in {'output', 'temp'}:
            continue
        relative = Path(descriptor.get("subfolder", "")) / descriptor["filename"]
        source = contained(proc[file_type], str(relative.parent)) / relative.name
        resolved = source.resolve()
        if not resolved.is_relative_to(Path(proc[file_type]).resolve()) and resolved not in {p.resolve() for p in previous.values()}:
            raise ValueError("Output escapes renderer and run roots")
        if str(source) in seen:
            continue
        seen.add(str(source))
        target = previous.get(str(source), path / (str(len(outputs)) + '-' + source.name))
        if source.resolve() != target.resolve():
            temporary = target.with_suffix(target.suffix + '.partial')
            before = signature(source)
            shutil.copyfile(source, temporary)
            if before != signature(source):
                raise ValueError('Output changed while archiving: ' + str(source))
            os.replace(temporary, target)
        outputs.append({'path': str(target), 'sha256': digest(target), 'bytes': target.stat().st_size, 'source': str(source)})
    changed = [a['path'] for a in manifest['assets'] if signature(a['path']) != a['signature']]
    if changed:
        manifest['gaps'].append('Assets changed during generation: ' + ', '.join(changed))
        manifest['provenance'] = 'incomplete'
    success = history.get('status', {}).get('status_str') == 'success'
    if success and not outputs:
        raise ValueError('Renderer completed but no output files were reported; archive remains incomplete')
    save(path, manifest, 'succeeded' if success else 'failed', outputs=outputs, completed_at=now(),
         error=None if success else 'Renderer failed; see history.json')

    for output in outputs:
        source = Path(output["source"])
        if source.is_relative_to(Path(proc["output"]) / "media-runs" / manifest["run_id"]):
            output_alias(source, Path(output["path"]), output["sha256"])
    if success:
        try:
            from library import frame_outputs, link_output
            frames = {o['path'] for o in frame_outputs(path, manifest)}
            has_video = any(Path(o['path']).suffix.lower() in {'.mp4', '.webm', '.mov'} for o in outputs)
            for output in outputs:
                if not has_video or output['path'] not in frames:
                    link_output(ROOT, manifest, output)
        except (OSError, ValueError, KeyError) as error:
            print('Media library update skipped: ' + str(error), file=sys.stderr)
        try:
            cleanup_staging(path, manifest, apply=True)
        except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
            print('Input staging cleanup skipped: ' + str(error), file=sys.stderr)


def execute(run_id, recover=False):
    path = run_path(run_id)
    with (path / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        manifest = read(path / 'manifest.json')
        if recover and manifest['state'] not in {'submitted', 'submitting', 'uncertain', 'archive_failed'}:
            return
        if not recover and manifest['state'] not in {'staged', 'prepared'}:
            return
        trace = MemoryTrace(path)
        trace.start()
        try:
            if recover:
                if manifest['state'] not in {'submitted', 'submitting', 'uncertain', 'archive_failed'}:
                    raise ValueError('Only submitted/uncertain/archive_failed runs can resume; this never resubmits')
            else:
                if manifest['state'] not in {'staged', 'prepared'}:
                    raise ValueError('Run already started; use resume rather than submitting twice')
                save(path, manifest, 'preparing')
                prepare(path, manifest)
                if queue_rows(manifest['backend']) and not manifest.get('allow_busy'):
                    raise ValueError('ComfyUI is busy. No submission made. Use a new submit with --allow-busy to append intentionally.')
                snapshot = read(path / 'runtime.json')
                if process(manifest['backend']) != snapshot['process']:
                    raise ValueError('Renderer process changed during preparation')
                for record in manifest['assets']:
                    if signature(record['path']) != record['signature']:
                        raise ValueError('Asset changed during preparation: ' + record['path'])
                payload = {'prompt': read(path / 'graph-api.json'), 'client_id': 'render-media:' + run_id,
                           'extra_data': {'media_render_run_id': run_id}}
                if (path / 'workflow-ui.json').exists():
                    payload['extra_data']['extra_pnginfo'] = {'workflow': read(path / 'workflow-ui.json')}
                save(path, manifest, 'submitting', submission_attempted_at=now())
                try:
                    response = api(manifest['backend'], '/prompt', payload)
                except urllib.error.HTTPError as error:
                    if error.code in {400, 422}:
                        (path / 'rejection.json').write_bytes(error.read())
                        save(path, manifest, 'failed', error='ComfyUI rejected graph; see rejection.json')
                        return
                    raise
                if not response.get('prompt_id'):
                    raise ValueError('Submission acknowledgement has no prompt ID')
                save(path, manifest, 'submitted', prompt_id=response['prompt_id'])
            failures = 0
            while True:
                try:
                    prompt_id, history = locate(manifest)
                    failures = 0
                except (OSError, ValueError):
                    failures += 1
                    if failures >= 20:
                        raise ValueError('Backend unreachable; use resume after inspecting the renderer')
                    time.sleep(3)
                    continue
                if not prompt_id:
                    save(path, manifest, 'uncertain', error='No matching queue/history entry. Never automatically resubmitted.')
                    return
                if manifest.get('prompt_id') != prompt_id or manifest['state'] != 'submitted':
                    save(path, manifest, 'submitted', prompt_id=prompt_id)
                if history:
                    try:
                        collect(path, manifest, history)
                    except Exception as error:
                        save(path, manifest, 'archive_failed', error=str(error))
                    return
                time.sleep(3)
        except Exception as error:
            state = 'uncertain' if manifest['state'] in {'submitting', 'submitted', 'uncertain'} else 'failed'
            save(path, manifest, state, error=str(error))
        finally:
            trace.stop()


def launch(run_id, recover=False):
    path = run_path(run_id)
    with (path / 'recorder.log').open('ab') as log:
        subprocess.Popen([sys.executable, str(HERE / 'recorder.py'), 'recover' if recover else 'execute', run_id],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    return public(read(path / 'manifest.json'))


def verify(path):
    m = read(path / 'manifest.json')
    problems = []
    records = m['assets'] + m['inputs'] + m['outputs']
    for record in records:
        try:
            if digest(record['path']) != record['sha256']:
                problems.append('Hash mismatch: ' + record['path'])
        except OSError:
            problems.append('Missing: ' + record['path'])
    for name, sha in [('graph-api.json', m.get('graph_sha256')), ('runtime.json', m.get('runtime_sha256'))]:
        if sha and (not (path / name).exists() or digest(path / name) != sha):
            problems.append('Changed archive: ' + name)
    if (path / 'runtime.json').exists():
        expected = read(path / 'runtime.json').get('source_archive_sha256')
        if expected and (not (path / 'source.tar.gz').exists() or digest(path / 'source.tar.gz') != expected):
            problems.append('Changed source archive')
    return {'run_id': m['run_id'], 'ok': not problems, 'problems': problems, 'state': m['state'], 'provenance': m.get('provenance'), 'gaps': m['gaps']}


def stack_identity(snapshot):
    return {k: snapshot[k] for k in ['packages', 'python', 'driver', 'sources']} | {'flags': snapshot['process']['flags'], 'environment': snapshot['process']['environment'], 'native_libraries': snapshot.get('native_libraries', [])}


def rerun(request):
    old = run_path(request['run_id'])
    m = read(old / 'manifest.json')
    check = verify(old)
    if not check['ok']:
        raise ValueError('Cannot rerun: ' + '; '.join(check['problems']))
    live = runtime(m['backend'])
    # Capture current source identity in a temporary directory; retain no temporary run.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        capture_code(Path(temp), live)
    previous = read(old / 'runtime.json')
    if stack_identity(previous) != stack_identity(live) and not request.get('allow_changed_stack'):
        raise ValueError('Runtime/source stack changed. Inspect runtime.json; --allow-changed-stack explicitly permits a variation.')
    graph = read(old / 'graph-api.json')
    uploads = []
    for i, record in enumerate(m['inputs']):
        name = str(i) + Path(record['path']).suffix
        graph_value(graph, record['binding'], '__UPLOAD__/' + name)
        uploads.append({'binding': record['binding'], 'name': name, 'sha256': record['sha256'], 'source': record['path']})
    request.update(graph=graph, workflow=read(old / 'workflow-ui.json') if (old / 'workflow-ui.json').exists() else None,
                   uploads=uploads, project=m['project'], title=m['title'], recipe=m['recipe'], backend=m['backend'],
                   requester='rerun', allow_incomplete=m.get('allow_incomplete', False), keep_frames=m.get('keep_frames', False))
    result = create(request)
    path = run_path(result['run_id'])
    new = read(path / 'manifest.json')
    save(path, new, parent_run_id=m['run_id'], changed_stack_allowed=bool(request.get('allow_changed_stack')))
    for upload in uploads:
        dedupe_input(shutil.copyfile(upload['source'], path / 'inputs' / upload['name']))
    return launch(result['run_id'])


def dispatch(request):
    action = request['action']
    if action == 'cleanup':
        from cleanup import clean
        return clean(bool(request.get('apply')))
    if action == 'cleanup-orphans':
        from orphan_cleanup import clean_orphans
        return clean_orphans(bool(request.get('apply')))
    if action == 'library':
        from library import index
        return index(ROOT, bool(request.get('apply')))
    if action == 'create':
        return create(request)
    if action == 'doctor':
        backends = {}
        for backend in PORTS:
            try:
                proc = process(backend)
                rows = queue_rows(backend)
                backends[backend] = {'reachable': True, 'port': PORTS[backend], 'jobs': len(rows), 'pid': proc['pid'], 'cwd': proc['cwd']}
            except Exception as error:
                backends[backend] = {'reachable': False, 'port': PORTS[backend], 'reason': str(error)}
        return {'host': os.uname().nodename, 'archive': str(ROOT), 'backends': backends}
    if action == 'list':
        runs = []
        for manifest in sorted((ROOT / 'runs').glob('*/*/*/*/manifest.json'), reverse=True):
            m = read(manifest)
            if not request.get('project') or m['project'] == request['project']:
                runs.append(public(m))
            if len(runs) >= max(1, min(request.get('limit', 20), 500)):
                break
        return runs
    if action == 'inventory':
        snapshot = runtime(request['backend'])
        if request['kind'] == 'models':
            return {key: [{'category': c, 'path': p} for c, p in sorted(value)] for key, value in sorted(model_index(snapshot).items())}
        proc = snapshot['process']
        flags = proc['flags']
        user = Path(flags[flags.index('--user-directory') + 1]) if '--user-directory' in flags else Path(proc['cwd']) / 'user'
        return [str(p) for p in user.glob('**/workflows/*.json')]
    if action == 'rerun':
        return rerun(request)
    path = run_path(request['run_id'])
    m = read(path / 'manifest.json')
    if action == 'show':
        return m
    if action == 'memory':
        from memory_trace import summarize
        return {'run_id': m['run_id'], **summarize(path)}
    if action == 'status':
        return public(m)
    if action == 'start':
        if m['state'] not in {'staged', 'prepared'}:
            raise ValueError('Already started; use resume')
        return launch(m['run_id'])
    if action == 'resume':
        if m['state'] not in {'submitted', 'submitting', 'uncertain', 'archive_failed'}:
            raise ValueError('Only existing submissions can resume; this never resubmits')
        return launch(m['run_id'], recover=True)
    if action == 'prepare':
        if m['state'] not in {'staged', 'prepared'}:
            raise ValueError('Run already started')
        with (path / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return public(prepare(path, m))
    if action == 'verify':
        return verify(path)
    raise ValueError('Unknown action')


if __name__ == '__main__':
    os.umask(0o077)
    if len(sys.argv) > 1 and sys.argv[1] in {'execute', 'recover'}:
        execute(sys.argv[2], recover=sys.argv[1] == 'recover')
    else:
        try:
            print(json.dumps(dispatch(json.load(sys.stdin))))
        except Exception as error:
            print(json.dumps({'__rpc_error__': str(error)}))
            sys.exit(0)
