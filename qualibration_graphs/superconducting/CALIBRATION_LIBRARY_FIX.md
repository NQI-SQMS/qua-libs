# Fix: QualibrationLibrary not seeing nodes in CZ_calibrations

## Symptom

`QualibrationLibrary.get_active_library()` only returned nodes from
`calibrations/1Q_calibrations/`. Nodes in `calibrations/CZ_calibrations/`
were missing from `library.nodes`, even though the files exist in the repo
and are correctly imported elsewhere.

## Root cause

`QualibrationLibrary` scans for node/graph files by calling
`QualibrationNode.scan_folder_for_instances(path)` /
`QualibrationGraph.scan_folder_for_instances(path)` on the single folder
given by `calibration_library.folder` in `~/.qualibrate/config.toml`. That
method does a plain `path.iterdir()` — no recursion into subdirectories.
Confirmed both empirically (a node file one level deep is invisible; the
same file directly in the configured folder is found) and against the
public source (`scan_folder_for_instances` in
[qualibration_node.py](https://github.com/qua-platform/qualibrate-core/blob/main/qualibrate/qualibration_node.py)).
There's no config flag for recursion or multiple folders in `qualibrate`'s
config schema.

This project splits calibration nodes across two sibling folders:

```
calibrations/
├── 1Q_calibrations/   (~120 files)
└── CZ_calibrations/   (~30 files)
```

Since `folder` can only point at one location, and neither folder alone
contains the full node set, one of the two was always going to be invisible.

A second wrinkle: `QualibrationLibrary.get_active_library()` — the idiom
used in every notebook and node file (`from qualibrate import
QualibrationLibrary; library = QualibrationLibrary.get_active_library()`)
— always constructs a plain `qualibrate.QualibrationLibrary` internally, no
matter what class you call it on. So a fix built as a `QualibrationLibrary`
subclass only helps code that explicitly instantiates that subclass; it
does nothing for the existing idiom everywhere else in this repo.

## Fix

[`quam_config/recursive_calibration_scan.py`](./quam_config/recursive_calibration_scan.py)
monkeypatches `QualibrationLibrary._scan` itself (same pattern as the
existing `hardware_batching_patch.py`) to call the real, unmodified
`_scan()` once for `calibration_library.folder` and once for each of its
immediate subfolders, merging the resulting `nodes`/`graphs`. All the
fragile internals (background-loading queue, load-state tracking,
placeholder creation) stay inside the untouched private `_scan()` — this
only orchestrates calling it multiple times, so it doesn't need to
reimplement anything qualibrate might change between versions, and it picks
up new subfolders automatically (nothing is hardcoded).

Patching the method directly on the class — rather than subclassing — means
**every** consumer benefits automatically: `get_active_library()`, a
directly-constructed `QualibrationLibrary(...)`, and the QUAlibrate
runner's config-driven construction all go through the same (now-patched)
`_scan()`. No notebook or node file needs to change.

`quam_config/__init__.py` imports this patch module (applying it once) and
re-exports `QualibrationLibrary` for `config.toml`'s `resolver` to point at.
The `resolver` isn't there to select a different class — it's the same
`qualibrate.QualibrationLibrary` either way — it's there so that resolving
the dotted path forces `quam_config` (and this patch) to be imported before
the QUAlibrate runner constructs its library. Without that, the runner has
no other reason to ever import `quam_config`, so the patch would never be
applied in that process.

This replaces two earlier approaches that were tried and superseded:
- A `sync_calibration_library.py` script that hardlinked every node file
  into a flat mirror directory, needing a re-run whenever files changed.
- A `QualibrationLibrary` subclass (`RecursiveQualibrationLibrary`) plus a
  second patch to pre-populate `get_active_library()`'s internal cache —
  worked, but needed two cooperating pieces where one plain monkeypatch is
  enough.

### Setup (do this once per machine)

1. Check `quam_config` is importable in the environment qualibrate runs in
   (it registers as a top-level package via this project's normal
   `pip install -e .` from `qualibration_graphs/superconducting/`, so this
   is usually already true — but confirm it, don't assume it):

   ```bash
   python -c "import quam_config; print(quam_config.__file__)"
   ```

   If this fails, run `pip install -e .` from
   `qualibration_graphs/superconducting/` in the same environment qualibrate
   uses, then re-check.
2. Edit `~/.qualibrate/config.toml` (create the
   `[qualibrate.calibration_library]` section if it doesn't exist yet):

   ```toml
   [qualibrate.calibration_library]
   folder = "/absolute/path/to/qua-libs/qualibration_graphs/superconducting/calibrations"
   resolver = "quam_config.QualibrationLibrary"
   ```

   `folder` must point at the **parent** `calibrations/` directory (not
   `1Q_calibrations` or `CZ_calibrations` directly) — use your own absolute
   path, forward or double-backslashes on Windows.
3. Restart any already-running QUAlibrate process (web app / composite
   runner) and any Jupyter kernel that already imported `quam_config`. Both
   read config / import modules once at startup, so anything already
   running keeps the old behavior until restarted.

No script to re-run afterwards — adding, removing, or renaming a `.py` file
under `calibrations/` (or adding a whole new sibling subfolder) is picked up
automatically the next time the library scans.

### Verification

```python
from qualibrate import QualibrationLibrary
library = QualibrationLibrary.get_active_library()

len(library.nodes)              # all nodes from every subfolder, e.g. 122
"30_cz_iswap_flux_bootstrap" in library.nodes   # True
```

(Requires `quam_config` to have been imported earlier in the same
process/notebook kernel — true for every existing notebook here, since they
all import `Quam` from `quam_config` for machine setup before running any
node.)

The QUAlibrate runner constructs its library the same way it would in a
running process — by resolving `resolver`/`folder` straight from
`config.toml`, without any code importing `quam_config` directly first
(that's the whole point of the `resolver` hook: forcing that import). This
can be verified without importing `quam_config` at all:

```python
import tomllib
from pathlib import Path
from qualibrate_config.models import QualibrateConfig

with open(Path.home() / ".qualibrate" / "config.toml", "rb") as f:
    raw = tomllib.load(f)

cl = QualibrateConfig(raw["qualibrate"]).calibration_library
resolver_cls = cl.resolver          # <class 'qualibrate.core.qualibration_library.QualibrationLibrary'> (now patched)
library = resolver_cls(library_folder=cl.folder, set_active=True)

len(library.nodes)                              # 122
"30_cz_iswap_flux_bootstrap" in library.nodes   # True
```

### Does this cover the QUAlibrate web GUI too?

Yes. The GUI never builds its own `QualibrationLibrary` — `qualibrate.app`
(the web backend) only handles project metadata (folder paths as strings
for the settings page). Node/graph listing and execution, both for the API
and for what the GUI displays, go through the `runner` service's
`get_cached_library(config)`, which constructs the library from
`config.resolver` + `config.folder` — the exact same construction path
verified just above. So once the runner process is restarted with the
current config, the GUI reflects the full node list (both subfolders) on
next page load — no separate fix needed for it.

Practically: restart the QUAlibrate runner (or the whole composite process,
since `spawn = true` restarts app/runner/qua_dashboards together), then
refresh the browser tab. No need to also restart `qualibrate-app` on its
own — it doesn't cache the node list itself.

## Notes

- This is a per-machine fix: `~/.qualibrate/config.toml` is local, not part
  of the git repo. Every teammate needs to make the two config edits above
  once on their own machine — same as any other local qualibrate setting
  (`project`, `storage.location`, etc. are already machine-local).
- `quam_config/recursive_calibration_scan.py` *is* part of the repo, so it
  travels with every clone/branch — nothing else to install or maintain.
- `qualibrate`'s core package (`QualibrationLibrary` itself) ships as a
  closed/compiled wheel from some version onward — its scanning logic can't
  be patched in place, which is why this fix monkeypatches the method at
  runtime instead of editing the installed package.
- **Any already-running Python process (a live Jupyter kernel, the
  QUAlibrate web app/runner) needs to be restarted** to pick this up — it
  already imported the old `quam_config/__init__.py` (without the patch)
  and/or read the old config, and Python won't re-run either just because
  the files on disk changed.
- This patches a private method (`QualibrationLibrary._scan`), verified
  against `qualibrate` 1.4.0. `pyproject.toml` only pins `qualibrate>=1.0.2`
  (no upper bound), so a machine with a different version could in
  principle have restructured `_scan` internally. If that ever happens, the
  patch fails **loudly** at import time — `QualibrationLibrary._scan` not
  existing raises an `AttributeError` as soon as `quam_config` is imported,
  it doesn't silently no-op. Check your version with
  `python -c "import importlib.metadata as m; print(m.version('qualibrate'))"`
  if you hit that.
