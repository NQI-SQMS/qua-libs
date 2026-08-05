"""Patch `QualibrationLibrary` to scan `calibrations/` recursively (one level deep).

`QualibrationLibrary._scan()` only looks directly inside the single folder given by
`calibration_library.folder` in `~/.qualibrate/config.toml` — it does a plain
`path.iterdir()`, no recursion (confirmed against the installed qualibrate-core, and
matches its public source: https://github.com/qua-platform/qualibrate-core/blob/main/qualibrate/qualibration_node.py).
This project splits nodes across sibling folders (`calibrations/1Q_calibrations/`,
`calibrations/CZ_calibrations/`, ...), so a plain scan of `folder` always misses whatever
isn't directly inside it.

Importing this module (done once, from quam_config/__init__.py) replaces
`QualibrationLibrary._scan` with a version that calls the real, unmodified `_scan()` once
for `calibration_library.folder` itself and once for each of its immediate subfolders,
merging the resulting `nodes`/`graphs`. All the fragile internals (background-loading
queue, load-state tracking, placeholder creation) stay inside the untouched private
`_scan()` — this only orchestrates calling it multiple times, so every consumer
(`QualibrationLibrary.get_active_library()`, direct construction, the QUAlibrate runner)
gets recursive scanning automatically, with no subclass and no notebook changes needed.

`config.toml`'s `calibration_library.resolver` should point at
`quam_config.QualibrationLibrary` (see quam_config/__init__.py) rather than the default
`qualibrate.QualibrationLibrary` -- not because a different class is needed, but so that
importing the resolver forces `quam_config` (and this patch) to be imported before the
QUAlibrate runner constructs its library. `folder` should point at the parent
`calibrations/` directory, not at `1Q_calibrations` or `CZ_calibrations` directly.
"""
from qualibrate import QualibrationLibrary

_original_scan = QualibrationLibrary._scan


def _recursive_scan(self) -> None:
    base = self._library_folder
    if base is None:
        _original_scan(self)
        return

    folders = [base] + sorted(p for p in base.iterdir() if p.is_dir())
    all_nodes: dict = {}
    all_graphs: dict = {}
    for folder in folders:
        self._library_folder = folder
        _original_scan(self)
        all_nodes.update(self.nodes)
        all_graphs.update(self.graphs)

    self._library_folder = base
    self.nodes.clear()
    self.nodes.update(all_nodes)
    self.graphs.clear()
    self.graphs.update(all_graphs)


QualibrationLibrary._scan = _recursive_scan
