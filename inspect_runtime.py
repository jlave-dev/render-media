"""Executed by the renderer's Python; imports path configuration, never torch."""
import importlib.metadata
import json
import os
import sys

request = json.load(sys.stdin)
sys.path.insert(0, request['cwd'])
from comfy import cli_args
cli_args.args = cli_args.parser.parse_args(request['flags'])
import folder_paths
from utils.extra_config import load_extra_path_config
default_config = os.path.join(request['cwd'], 'extra_model_paths.yaml')
if os.path.isfile(default_config):
    load_extra_path_config(default_config)
for group in cli_args.args.extra_model_paths_config or []:
    for path in group:
        load_extra_path_config(path)
print(json.dumps({
    'packages': sorted(({'name': d.metadata['Name'], 'version': d.version} for d in importlib.metadata.distributions()), key=lambda x: x['name'].lower()),
    'model_paths': {k: v[0] for k, v in folder_paths.folder_names_and_paths.items() if k != 'custom_nodes'},
    'python': sys.version,
}))
