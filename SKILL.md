---
name: render-media
description: Submit ad hoc ComfyUI image or video renders to a remote Linux host with archived graphs, inputs, asset hashes, runtime records, and outputs. Use for new renders and rerunning or locating recorded generations.
---

# Render media

Run `render-media` on the client machine. Configure the SSH host and remote recorder path as described in the README. The authoritative archive lives on the remote host at `~/media/runs/YYYY/MM/DD/<run-id>/`. The machines do not need a shared filesystem.

Use this command for ad hoc ComfyUI submissions instead of raw `POST /prompt` or scripts that scatter originals. The CLI collects outputs into the archive, even when the submitting agent exits. Browser renders and scripts bypassing this CLI are not automatically captured.

## Submit

Start with `render-media doctor`. It reports live backend processes and job counts without dumping prompts. It never starts/stops services, clears queues, unloads models, or reserves a GPU. Check workload ownership/memory and the existing task's permission before loading a different model. An idle ComfyUI queue does not authorize evicting other workloads. Do not start a backend owned by the nightly queue or a distributed worker without coordinating with that owner.

Prepare an executable API graph as a local client JSON file. A visual UI workflow alone is not executable. Use `render-media inventory workflows` and `render-media inventory models` to discover available files; these inspect the selected live backend. Read a listed remote workflow with SSH when needed. Node availability and paths belong to that backend, not to the caller's checkout.

```sh
render-media submit /path/to/graph-api.json \
  --project portrait-tests --title "Window lighting variation" \
  --recipe qwen-image-experiment-v1 --wait
```

Use stable lowercase project slugs. Supply the editable graph with `--workflow workflow-ui.json` when available; it is archived separately and embedded as workflow metadata. The exact executable graph is always preserved. `--recipe` is a descriptive label, not a recipe installer or resolver.

For client input files and graph overrides:

```sh
render-media submit edit-api.json --project portrait-tests --title "Reference edit" \
  --input '11.image=/path/to/reference.png' \
  --set '8.seed=424242' --set '5.prompt="Soft window light"' --wait
```

`--input NODE.FIELD=LOCAL_FILE` streams a local file over SCP, verifies its hash, preserves it, and updates that graph input. Existing standard ComfyUI LoadImage/Video/Audio inputs on the remote host are copied into the run as well. `--set` values must be valid JSON; node fields are top-level input keys, not nested property expressions. Save-node filename prefixes are rewritten under `media-runs/<run-id>/`; original graph values remain in `graph-original.json`.

Defaults: backend `comfy` on remote host port 8188. `--backend image` selects 8196; `--backend dual` selects 8195. These must already be running. The CLI does not own service lifecycle, GPU scheduling, or distributed worker setup.

`--prepare-only` records and validates without posting a render. By default an occupied backend is refused. Add `--allow-busy` only when intentionally appending behind the existing jobs. Snapshotting and hashing can take time on the first use of large weights; later calls cache hashes using device/inode/size/mtime/ctime.

Unresolved assets, dynamic/external nodes, or a distributed worker environment are listed as provenance gaps and block submission. Inspect `render-media show <run-id>`. Use `--allow-incomplete` only when the requested experiment requires that unsupported path; disclose the remaining gaps. Never describe those runs as fully reproducible. The recorder captures the coordinator, not the distributed worker environment.

## Find, collect, and rerun

```sh
render-media list --project portrait-tests
render-media status <run-id>
render-media wait <run-id> --timeout 7200
render-media show <run-id>
render-media verify <run-id>
render-media rerun <run-id> --wait
render-media fetch <run-id> /path/to/render-copies
```

Without `--wait`, submission returns while a detached remote host recorder completes the run. A wait timeout does not cancel rendering. If the recorder crashes or loses backend access, inspect status and use `render-media resume <run-id>` to reconcile/collect an existing submission. Resume never posts another prompt. An uncertain acknowledgement is never automatically retried; inspect the retained backend queue/history before deliberately creating a replacement run.

Rerun verifies assets, inputs, output/archive integrity, and the live runtime/source identity before making a new run linked to its parent. A changed software stack blocks the rerun; `--allow-changed-stack` explicitly accepts a variation and records that choice. Same seeds/settings do not guarantee identical GPU pixels. Models are hashed but not duplicated into every run; retain/back up referenced weights. Runtime wheels are identified by package versions, not bundled for reinstall.

Return the run ID, backend status, and **remote host** canonical output paths. If a project needs local files, use `fetch` and identify them as copies. Do not call a run complete until status is `succeeded`; report `failed`, `uncertain`, `archive_failed`, and incomplete provenance accurately.

## Canonical media storage

remote host ~/media/runs/YYYY/MM/DD/<run-id>/ is the source of truth. After output collection is recorded and both hashes match, renderer output/media-runs paths become symlinks to the canonical archive files. They exist only for ComfyUI history compatibility; do not move or delete canonical outputs through those aliases. A collection failure retains the original. Fetch creates explicitly requested copies. Recovered outputs without completed manifests live in the matching run/recovered-outputs with recovery.json and remain unverified.

## Archive cleanup

`render-media cleanup` is a dry run; `render-media cleanup --apply` deduplicates verified inputs of successful, unlocked runs and removes safe staging copies. It leaves outputs and run records intact, checks all running ComfyUI queues before cleaning a retired backend, skips unsafe cases, and reports other run states for review. Staging cleanup can make inputs unavailable to old ComfyUI history; recorder reruns use the archive.
