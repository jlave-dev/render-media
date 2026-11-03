# render-media

Record ComfyUI image and video runs on a remote Linux machine. This command submits an API graph over SSH and keeps the graph, inputs, output files, asset hashes, and runtime details together for later inspection. It can verify a recorded run, recover an interrupted collection, and attempt a rerun.

This tool was built for a two-DGX-Spark setup. The remote recorder uses Linux process inspection and file locking and expects an NVIDIA driver utility. It uses the backend ports listed below. Other machines need a compatible ComfyUI setup.

The repository includes two small CPU example graphs and a test suite. The examples require a running, compatible ComfyUI backend. A UI workflow file alone is not an executable API graph. See `SKILL.md` for more commands.

## Setup

Install Python 3.9 or newer on both machines, and SSH and SCP on the client. Configure key-based SSH access to the remote Linux machine. The remote machine needs a running ComfyUI process, its Python environment, and `nvidia-smi`. From a checkout on the client, copy the Python files to the remote machine:

```sh
ssh your-host 'mkdir -p ~/dev/render-media'
scp ./*.py your-host:dev/render-media/
export RENDER_MEDIA_HOST=your-host
./render.py doctor
```

Use the SSH host name or alias in place of `your-host`. By default the client looks for `~/dev/render-media/recorder.py` in the remote user's home and connects to `spark-2`. Set `RENDER_MEDIA_HOST` or pass `--host` to select another host. Set `RENDER_MEDIA_REMOTE_PATH` or pass `--remote-path` to select another remote recorder location; use an absolute path or one beginning with `~/`. The command-line options take precedence over the environment. Both options go before the subcommand. Deploy all Python modules together when updating the remote recorder.

For example, with a compatible ComfyUI backend running on port 8188, prepare the included CPU graph, inspect its reported provenance gaps, and then submit it:

```sh
./render.py submit examples/cpu-smoke.json --project smoke --title 'CPU smoke test' --prepare-only
./render.py show RUN_ID
./render.py submit examples/cpu-smoke.json --project smoke --title 'CPU smoke test' --wait
```

Preparation creates an archive entry but does not run the graph. If it reports provenance gaps, inspect them and use `--allow-incomplete` only if you accept them. Archives are stored under `~/media/runs/` on the remote machine. You can add a `render-media` symlink to `render.py` on your PATH; examples below use that name.

Python's standard library supplies SSH orchestration, HTTP, hashing, archives, locks, and the CLI. The renderer's existing Python interpreter is used only to inspect installed package metadata and ComfyUI path configuration; no new dependency or model is installed and torch is not imported by the inspection helper.

`render-media --help` lists commands. `submit --prepare-only` does not submit work. Normal submit returns a run ID and starts an independent recorder, with optional `--wait`. Every run retains its original graph, resolved API graph, optional editable workflow, inputs, asset hashes, package/runtime/process information, backend node schemas, source archive, execution history, and collected outputs. The recorder samples system memory every 50 ms while it executes or recovers a run, writing `memory-samples.csv` in that run; collection failures are reported in `recorder.log` and do not stop a render. Files are private by default. Models remain in the shared model library. `verify` rehashes recorded files; `rerun` checks the saved stack before submitting a new linked run.

The recorder changes only filename prefixes and explicitly uploaded/overridden graph inputs. It never starts/stops services, unloads models, interrupts work, or manages GPU leases. Busy submission requires `--allow-busy`. Unresolved provenance requires `--allow-incomplete`. The three configured backends are `comfy` (8188), `image` (8196), and `dual` (8195), all accessed on the remote host over SSH. Different ports currently require editing `PORTS` in `recorder.py`. Only coordinator provenance is currently captured for distributed rendering.

This is CLI enforcement, not a server interception layer. Raw API/browser submissions and arbitrary standalone scripts bypass it. Archive copies are collected from backend history; custom save nodes must report their files there. Known dynamic/external nodes are flagged, but arbitrary node implementations cannot be proven self-contained. Source snapshots record disk contents, not Python modules already imported before a disk edit. Source snapshots omit symlinks, model files, credential files, and individual files larger than 16 MiB. Installed binary wheels are not bundled. Capture records and backup retention are distinct responsibilities. Seed/stack equality does not guarantee identical pixels.

Recovery: inspect `show`, `status`, `recorder.log`, and `history.json` inside the run. `resume` reconciles an existing submission and recollects outputs without reposting. Missing acknowledgements/history become `uncertain` and are never automatically rerendered. `fetch` copies a whole run to the client without moving originals.

Validation: `python3 -m unittest -v test_recorder.py`. Tests use a real local HTTP fixture and temporary files, without GPU work.

Live verification on 2026-10-23: 12 tests passed on the original client. Real client-to-remote CPU-only submission (`20261023T071041Z-68b07ce0e2ae`), rerun (`20261023T071147Z-13ee905c3814`), and local-input upload (`20261023T071217Z-97d992900a47`) all completed and archived. All three 64x64 images decoded to identical RGB pixels. Prepare-only inspection (`20261023T071348Z-e2b41fc4ee6a`) captured a real installed VAE hash without executing it. Fetch and model inventory also passed. These checks do not qualify GPU model quality, video encoding, or distributed worker provenance.

## Canonical media storage

The remote host’s ~/media/runs/YYYY/MM/DD/<run-id>/ is the source of truth. After output collection is recorded and both hashes match, renderer output/media-runs paths become symlinks to the canonical archive files. They exist only for ComfyUI history compatibility; do not move or delete canonical outputs through those aliases. A collection failure retains the original. Fetch creates explicitly requested copies. Recovered outputs without completed manifests live in the matching run/recovered-outputs with recovery.json and remain unverified.

Browse finished media on the remote host in `~/media/library/YYYY/MM/DD/`. Filenames start with the run's UTC date and time followed by the description; a number distinguishes collisions. These are hard links to the canonical outputs, so they do not duplicate media bytes or change recorded paths. New runs put outputs directly in the run directory; existing output paths remain unchanged. `render-media library` previews links and reports recognizable saved video-frame sequences; `render-media library --apply` creates links for succeeded runs. Neither command deletes frames. It excludes matching PNG frame sequences when their run also contains a video. Failed and incomplete runs are skipped.

New submissions omit a `SaveImage` node only when it saves an `ImageFromBatch` sequence from the same image batch as a saved `CreateVideo`. The original graph remains recorded; `graph-api.json` contains the submitted graph and the manifest lists `omitted_frame_save_nodes`. Use `submit --keep-frames` to retain those exports. Other image saves are preserved.

## Archive cleanup

`render-media cleanup-orphans` previews successful runs for which no output remains linked in the media library. Add `--apply` to permanently remove those run directories and their matching recorder-owned ComfyUI history symlinks. It preserves runs with any remaining library link, including renamed links, and skips locked, incomplete, missing, modified, or unexpected files. It refuses to run when the library is missing. Run this before `library --apply` when missing links were deliberately deleted, since that command recreates library entries.

`render-media cleanup` reports input deduplication and safely removable staging copies without changing files. Add `--apply` to deduplicate inputs of successful, unlocked runs and remove verified staging copies when the backend queue can be checked, or when a retired backend is unused and all running ComfyUI queues are idle. It does not delete runs or outputs. Failed, prepared, staged, uncertain, and archive-failed runs are listed for review. Inputs are skipped when running ComfyUI queues cannot be checked or file hashes do not match. Saved run input paths remain valid, while old ComfyUI history may no longer find removed staging inputs; use the recorder archive to rerun.

Future successful renders clean up eligible staging inputs and share archived inputs by content. Studio's default `extra_model_paths.yaml` is included in model-path discovery.
