#!/usr/bin/env python3
"""Submit and inspect remote ComfyUI runs over SSH."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from common import digest, extension, graph_value, read

REMOTE = '~/dev/render-media/recorder.py'


def remote_path(path):
    """Quote a remote path while letting the remote shell resolve its home."""
    if path.startswith('~/'):
        return '"$HOME"/' + shlex.quote(path[2:])
    if not path.startswith('/'):
        raise ValueError('Remote recorder path must be absolute or start with ~/')
    return shlex.quote(path)


def rpc(args, action, **payload):
    payload.update(action=action)
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', args.host,
         'python3 ' + remote_path(getattr(args, 'remote_path', REMOTE)) + ' rpc'],
        input=json.dumps(payload), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Remote recorder failed')
    result = json.loads(result.stdout)
    if isinstance(result, dict) and '__rpc_error__' in result:
        raise RuntimeError(result['__rpc_error__'])
    return result


def transfer(args, source, target):
    # Archive paths and generated names never contain shell metacharacters.
    remote = args.host + ':' + target
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '--',
                    str(Path(source).resolve()), remote], check=True)


def wait(args, run_id):
    deadline = time.monotonic() + args.timeout
    previous = None
    while True:
        result = rpc(args, 'status', run_id=run_id)
        if result['state'] != previous:
            print(run_id + ': ' + result['state'], file=sys.stderr)
            previous = result['state']
        if result['state'] in {'succeeded', 'failed', 'uncertain', 'archive_failed'}:
            return result
        if time.monotonic() >= deadline:
            raise RuntimeError('Wait timed out; remote recording continues. Use render-media status ' + run_id)
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(prog='render-media', description=__doc__)
    parser.add_argument('--host', default=os.environ.get('RENDER_MEDIA_HOST', 'spark-2'),
                        help='SSH host (default: RENDER_MEDIA_HOST or spark-2)')
    parser.add_argument('--remote-path', default=os.environ.get('RENDER_MEDIA_REMOTE_PATH', REMOTE),
                        help='Remote recorder path (default: RENDER_MEDIA_REMOTE_PATH or ~/dev/render-media/recorder.py)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Read-only backend/installation checks')
    p = sub.add_parser('submit', help='Archive and submit a ComfyUI API graph immediately')
    p.add_argument('graph', type=Path)
    p.add_argument('--project', required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--recipe', default='ad-hoc', help='Recipe name/revision or experiment label')
    p.add_argument('--backend', choices=['comfy', 'image', 'dual'], default='comfy')
    p.add_argument('--workflow', type=Path, help='Optional editable UI workflow JSON')
    p.add_argument('--input', action='append', default=[], metavar='NODE.FIELD=LOCAL_FILE')
    p.add_argument('--set', action='append', default=[], metavar='NODE.FIELD=JSON_VALUE')
    p.add_argument('--allow-busy', action='store_true', help='Explicitly append behind existing ComfyUI jobs')
    p.add_argument('--allow-incomplete', action='store_true', help='Explicitly accept listed provenance gaps')
    p.add_argument('--keep-frames', action='store_true', help='Keep recognized video-frame PNG exports')
    p.add_argument('--prepare-only', action='store_true', help='Record/validate without submitting or starting services')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--timeout', type=int, default=3600)
    for name in ['status', 'show', 'verify', 'resume', 'wait', 'fetch', 'memory']:
        p = sub.add_parser(name)
        p.add_argument('run_id')
        if name == 'wait':
            p.add_argument('--timeout', type=int, default=3600)
        if name == 'fetch':
            p.add_argument('directory', type=Path)
    p = sub.add_parser('list')
    p.add_argument('--project')
    p.add_argument('--limit', type=int, default=20)
    p = sub.add_parser('inventory', help='List backend models or saved workflows')
    p.add_argument('kind', choices=['models', 'workflows'])
    p.add_argument('--backend', choices=['comfy', 'image', 'dual'], default='comfy')
    p = sub.add_parser('cleanup', help='Report or deduplicate finished-run inputs and remove safe staging copies')
    p.add_argument('--apply', action='store_true', help='Apply cleanup; default is a dry run')
    p = sub.add_parser('cleanup-orphans', help='Remove completed runs whose outputs no longer appear in the media library')
    p.add_argument('--apply', action='store_true', help='Delete eligible runs; default is a dry run')
    p = sub.add_parser('library', help='Preview or add finished media to the browseable library')
    p.add_argument('--apply', action='store_true', help='Create hard links; default is a dry run')
    p = sub.add_parser('rerun', help='Verify saved stack and create a new run from saved graph/inputs')
    p.add_argument('run_id')
    p.add_argument('--allow-changed-stack', action='store_true')
    p.add_argument('--allow-busy', action='store_true')
    p.add_argument('--wait', action='store_true')
    p.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.command == 'submit':
        graph = read(args.graph)
        if 'prompt' in graph and isinstance(graph['prompt'], dict):
            graph = graph['prompt']
        if not isinstance(graph, dict) or not graph or any(not isinstance(n, dict) or 'class_type' not in n or not isinstance(n.get('inputs'), dict) for n in graph.values()):
            raise ValueError('Expected an executable ComfyUI API graph, not a UI workflow')
        for setting in args.set:
            binding, value = setting.split('=', 1)
            graph_value(graph, binding, json.loads(value))
        uploads = []
        for i, binding in enumerate(args.input):
            key, source = binding.split('=', 1)
            source = Path(source).expanduser().resolve(strict=True)
            name = str(i) + extension(source)
            graph_value(graph, key, '__UPLOAD__/' + name)
            uploads.append({'binding': key, 'name': name, 'sha256': digest(source), 'source': str(source)})
        result = rpc(args, 'create', graph=graph, workflow=read(args.workflow) if args.workflow else None,
                     project=args.project, title=args.title, recipe=args.recipe, backend=args.backend,
                     uploads=uploads, requester=os.environ.get('USER', 'agent'),
                     allow_busy=args.allow_busy, allow_incomplete=args.allow_incomplete, keep_frames=args.keep_frames)
        print('Created ' + result['run_id'], file=sys.stderr)
        for upload in uploads:
            transfer(args, upload['source'], result['path'] + '/inputs/' + upload['name'])
        result = rpc(args, 'prepare' if args.prepare_only else 'start', run_id=result['run_id'])
        if args.wait and not args.prepare_only:
            result = wait(args, result['run_id'])
    elif args.command == 'wait':
        result = wait(args, args.run_id)
    elif args.command == 'fetch':
        result = rpc(args, 'show', run_id=args.run_id)
        args.directory.mkdir(parents=True, exist_ok=True)
        destination = args.directory / args.run_id
        if destination.exists():
            raise ValueError('Destination already exists: ' + str(destination))
        subprocess.run(['scp', '-qr', '-o', 'BatchMode=yes', '--', args.host + ':' + result['path'], str(args.directory.resolve())], check=True)
        result = {'run_id': args.run_id, 'local_path': str(destination.resolve())}
    else:
        payload = {k: v for k, v in vars(args).items() if k not in {'host', 'remote_path', 'command', 'wait', 'timeout'}}
        result = rpc(args, args.command, **payload)
        if args.command == 'rerun' and args.wait:
            result = wait(args, result['run_id'])
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and (result.get('ok') is False or (args.command not in {'verify', 'show'} and result.get('state') in {'failed', 'uncertain', 'archive_failed'})):
        return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError, KeyError) as error:
        print('render-media: ' + str(error), file=sys.stderr)
        sys.exit(1)
