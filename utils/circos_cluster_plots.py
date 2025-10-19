#!/usr/bin/env python3
"""Regenerate blockmodel plots for ARACNe cluster networks.

This script reuses the stochastic block model utilities originally developed in
``nb_graphAnalysis/sbm_cluster.ipynb`` to load ARACNe-derived graphs and render
the default blockmodel layout. It expects the Pearson-annotated network files
produced by ``postprocess_cluster_networks.py`` inside
``outputARACNE/cluster_runs/<stage>/cluster_XX`` directories.
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import textwrap

import graph_tool.all as gt  # type: ignore[import]


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NET_ROOT = REPO_ROOT / "outputARACNE"
DEFAULT_CLUSTER_ROOT = DEFAULT_NET_ROOT / "cluster_runs"
METADATA_DIR = REPO_ROOT / "metadata"


@dataclass(frozen=True)
class StageConfig:
    name: str
    time_label: str


STAGE_CONFIGS: Dict[str, StageConfig] = {
    "early": StageConfig(name="early", time_label="I-II_leidenwu"),
    "late": StageConfig(name="late", time_label="III-IV_leidenwu"),
}


# ---------------------------------------------------------------------------
# Metadata helpers (adapted from sbm_cluster.ipynb)
# ---------------------------------------------------------------------------

def _locate_metadata(filename: str = "cell_mappings.json") -> Path:
    search_path = Path.cwd()
    for candidate_dir in [search_path, *search_path.parents]:
        candidate = candidate_dir / "metadata" / filename
        if candidate.exists():
            return candidate
    fallback = METADATA_DIR / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find {filename} starting from {search_path}")


with _locate_metadata().open() as fh:
    METADATA = json.load(fh)

COLOR_MAP: Dict[str, str] = dict(METADATA.get("color_map", {}))
COLOR_MAP.setdefault("unknown", "#bcbcbc")
CELL_CATEGORIES = METADATA.get("cell_categories", {})
CELL_TYPE_TO_CATEGORY = {
    cell_type: category
    for category, cell_types in CELL_CATEGORIES.items()
    for cell_type in cell_types
}


def cell_category_mapping(cell_type: str) -> str:
    if "Tumor" in cell_type:
        return "tumoral"
    return CELL_TYPE_TO_CATEGORY.get(cell_type, "unknown")


# ---------------------------------------------------------------------------
# Graph helpers copied and adapted from sbm_cluster.ipynb
# ---------------------------------------------------------------------------

def get_state_from_file(
    time: str,
    *,
    net_folder: Path | str = DEFAULT_NET_ROOT,
    is_networkx: bool = False,
    remove_neg: bool = False,
    state_cache: Path | None = None,
    force: bool = False,
) -> Tuple[gt.NestedBlockState, gt.Graph]:
    """Load a graph and compute its blockmodel state from CSV exports."""

    if isinstance(net_folder, Path):
        folder = str(net_folder)
    else:
        folder = net_folder

    txt_simple_candidates = [
        os.path.join(folder, f"net{time}.txt"),
        os.path.join(folder, f"net_{time}.txt"),
    ]
    txt_nx_candidates = [
        os.path.join(folder, f"net{time}_MI_pearson.txt"),
        os.path.join(folder, f"net_{time}_MI_pearson.txt"),
    ]

    def _first_existing(paths: Iterable[str]) -> Optional[str]:
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    txt_simple = _first_existing(txt_simple_candidates)
    txt_nxlike = _first_existing(txt_nx_candidates)

    if not is_networkx:
        if txt_simple is None:
            raise FileNotFoundError(f"Could not locate CSV for time '{time}' in {folder}")
        file_path = txt_simple
        g = gt.load_graph_from_csv(
            file_path,
            csv_options={"delimiter": "\t"},
            eprop_types=["float"],
            eprop_names=["MI"],
        )
    else:
        if txt_nxlike is None:
            raise FileNotFoundError(f"Could not locate MI/Pearson CSV for time '{time}' in {folder}")
        file_path = txt_nxlike
        g = gt.load_graph_from_csv(
            file_path,
            csv_options={"delimiter": "\t"},
            eprop_types=["float", "float", "float", "float", "string"],
            eprop_names=["MI", "pvals", "pearson", "pvalue", "sign"],
        )

    edges = list(g.edges())
    seen: set[Tuple[int, int]] = set()
    for e in edges:
        src = int(e.source())
        dst = int(e.target())
        if (src, dst) in seen or (dst, src) in seen:
            g.remove_edge(e)
        else:
            seen.add((src, dst))

    if remove_neg and "sign" in g.ep:
        for e in list(g.edges()):
            if g.ep.sign[e] == "negative_significant":
                g.remove_edge(e)

    LOGGER.info("Graph loaded: %s", g)

    state: gt.NestedBlockState | None = None
    if state_cache and not force and state_cache.exists():
        try:
            LOGGER.info("Loading cached block state: %s", state_cache)
            with state_cache.open("rb") as fh:
                payload = pickle.load(fh)
            if isinstance(payload, dict):
                bs_data = payload.get("bs")
                bs_arrays: Optional[List[np.ndarray]] = None
                if bs_data is not None:
                    bs_arrays = []
                    for level in bs_data:
                        if hasattr(level, "a"):
                            raw = level.a
                        else:
                            raw = level
                        if isinstance(raw, np.ndarray):
                            arr = raw.astype(np.int64, copy=False)
                        else:
                            try:
                                arr = np.asarray(raw, dtype=np.int64)
                            except (TypeError, ValueError):
                                arr = np.fromiter(raw, dtype=np.int64)
                        bs_arrays.append(arr)
                state = gt.NestedBlockState(g, bs=bs_arrays)
            elif isinstance(payload, tuple) and len(payload) == 2:
                args, kwargs = payload
                state = gt.NestedBlockState(g, *args, **kwargs)
            else:
                raise TypeError(f"Unsupported cache payload type: {type(payload)}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load cached state %s: %s", state_cache, exc)
            state = None

    if state is None:
        LOGGER.info("Computing new block state for %s", time)
        new_state = gt.minimize_nested_blockmodel_dl(g)
        entropy_plain = new_state.entropy()
        entropy_annealed = gt.mcmc_anneal(
            new_state,
            beta_range=(1, 20),
            niter=1500,
            mcmc_equilibrate_args=dict(force_niter=20),
        )
        LOGGER.info("Entropy (plain): %s", entropy_plain)
        LOGGER.info("Entropy (annealed): %s", entropy_annealed)
        if state_cache:
            try:
                bs_serialized: List[List[int]] = []
                for level in new_state.get_bs():
                    if hasattr(level, "a"):
                        arr = np.asarray(level.a, dtype=np.int64)
                    else:
                        try:
                            arr = np.asarray(level, dtype=np.int64)
                        except (TypeError, ValueError):
                            arr = np.fromiter(level, dtype=np.int64)
                    bs_serialized.append(arr.tolist())

                cache_payload = {"bs": bs_serialized}
                with state_cache.open("wb") as fh:
                    pickle.dump(cache_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
                LOGGER.info("Saved block state cache to %s", state_cache)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to save block state cache %s: %s", state_cache, exc)
        state = new_state
    else:
        LOGGER.info("Loaded cached block state for %s", time)

    return state, g


# ---------------------------------------------------------------------------
# Rendering utilities
# ---------------------------------------------------------------------------


def paint_net_from_state(
    state: gt.NestedBlockState,
    g: gt.Graph,
    time: str,
    *,
    width_divisor: float = 3.0,
    margin: float = 0.25,
    remove_neg: bool = False,
    output_dir: Path | str | None = None,
    mi_threshold: float = 0.0,
) -> Path:
    if output_dir is None:
        output_dir = DEFAULT_NET_ROOT
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    g.vp["cat"] = g.new_vertex_property("string")
    g.vp["color"] = g.new_vertex_property("string")
    g.vp["label"] = g.new_vertex_property("string")
    for v in g.vertices():
        name = g.vp.name[v] if "name" in g.vp else str(int(v))
        category = cell_category_mapping(name)
        color = COLOR_MAP.get(category, COLOR_MAP["unknown"])
        g.vp.cat[v] = category
        g.vp.color[v] = color
        wrapped = "\n".join(textwrap.wrap(name, width=10)) if len(name) > 15 else name
        g.vp.label[v] = name
    
    edge_width = g.new_edge_property("float")
    if "MI" in g.ep:
        adjusted = np.maximum(g.ep.MI.a - mi_threshold, 0.0)
        edge_width.a = adjusted / width_divisor
    elif "weight" in g.ep:
        edge_width.a = g.ep.weight.a / width_divisor
    else:
        edge_width.a = np.full(g.num_edges(), 1.0 / width_divisor)

    edge_dash = g.new_edge_property("vector<float>")
    if "sign" in g.ep:
        for e in g.edges():
            sign = g.ep.sign[e]
            if sign == "positive_significant":
                edge_dash[e] = [3, 3, 3]
            elif sign == "negative_significant":
                edge_dash[e] = [10, 10, 10]
            else:
                edge_dash[e] = [1, 0, 0]

    edge_gradient = g.new_edge_property("vector<float>")
    for e in g.edges():
        src = e.source()
        rgba = list(mcolors.to_rgba(g.vp.color[src]))
        edge_gradient[e] = [0, rgba[0], rgba[1], rgba[2], 0.7, 1, rgba[0], rgba[1], rgba[2], 0.2]

    fig, ax = plt.subplots(figsize=(12, 10))
    drawing = state.draw(
        vertex_fill_color=g.vp.color,
        edge_pen_width=edge_width,
        vertex_text=g.vp.label,
        vertex_font_size=0.05,
        edge_dash_style=edge_dash,
        vertex_text_position="centered",
        edge_gradient=edge_gradient,
        vertex_pen_width=0.0,
        mplfig=ax,
    )
    drawing[0].fit_view(margin=margin, yflip=True)
    ax.set_title(time)
    ax.set_axis_off()

    handles: List[mpatches.Patch | Line2D] = []
    for category, color in COLOR_MAP.items():
        if category == "unknown":
            continue
        handles.append(mpatches.Patch(color=color, label=category))
    if "sign" in g.ep and not remove_neg:
        handles.append(Line2D([], [], color="black", linestyle="--", label="negative_significant"))
    handles.append(Line2D([], [], color="black", linestyle="dotted", label="positive_significant"))
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1, 0.5), loc="upper left")
    plt.subplots_adjust(right=0.8, left=0.01, bottom=0.01, top=0.95)

    out_file = output_path / f"graph_{time}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate blockmodel plots for ARACNe cluster networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(STAGE_CONFIGS.keys()),
        default=sorted(STAGE_CONFIGS.keys()),
        help="Stage folders to process",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        help="Optional list of cluster ids to process",
    )
    parser.add_argument(
        "--cluster-root",
        type=Path,
        default=DEFAULT_CLUSTER_ROOT,
        help="Location of the per-stage cluster directories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute block states even if a cached version exists",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def _iter_clusters(stage_dir: Path) -> Iterator[Tuple[int, Path]]:
    for child in sorted(stage_dir.iterdir()):
        if child.is_dir() and child.name.startswith("cluster_"):
            try:
                cluster_id = int(child.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            yield cluster_id, child


def _filter_clusters(all_clusters: Iterable[int], selection: Optional[Sequence[int]]) -> List[int]:
    clusters = sorted(set(all_clusters))
    if selection is None:
        return clusters
    wanted = set(selection)
    return [cid for cid in clusters if cid in wanted]


def _load_mi_threshold(cluster_dir: Path) -> float:
    threshold_files = sorted(cluster_dir.glob("miThreshold_*.txt"))
    if not threshold_files:
        LOGGER.debug("No MI threshold file found in %s; defaulting to 0", cluster_dir)
        return 0.0
    threshold_path = threshold_files[0]
    try:
        raw_value = float(threshold_path.read_text().strip())
        rounded = math.floor(raw_value * 10.0) / 10.0
        LOGGER.info(
            "Using MI threshold %.3f (rounded down to %.1f) from %s",
            raw_value,
            rounded,
            threshold_path,
        )
        return rounded
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read MI threshold from %s: %s", threshold_path, exc)
        return 0.0


def process_cluster(
    stage: StageConfig,
    cluster_id: int,
    cluster_dir: Path,
    *,
    force: bool,
) -> None:
    time_key = f"{stage.time_label}_cluster{cluster_id}"
    LOGGER.info("Processing stage=%s cluster=%02d", stage.name, cluster_id)

    state_cache = cluster_dir / f"state_{time_key}.pkl"

    try:
        state, g = get_state_from_file(
            time_key,
            net_folder=cluster_dir,
            is_networkx=True,
            state_cache=state_cache,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load graph for stage=%s cluster=%02d: %s", stage.name, cluster_id, exc)
        return

    try:
        mi_threshold = _load_mi_threshold(cluster_dir)
        paint_net_from_state(
            state,
            g,
            time_key,
            output_dir=cluster_dir,
            mi_threshold=mi_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not render blockmodel plot for stage=%s cluster=%02d: %s", stage.name, cluster_id, exc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    for stage_name in args.stages:
        stage = STAGE_CONFIGS[stage_name]
        stage_dir = args.cluster_root / stage.name
        if not stage_dir.exists():
            LOGGER.warning("Stage directory not found: %s", stage_dir)
            continue

        all_clusters = [cid for cid, _ in _iter_clusters(stage_dir)]
        selected_clusters = _filter_clusters(all_clusters, args.clusters)
        if not selected_clusters:
            LOGGER.warning("No clusters selected for stage %s", stage.name)
            continue

        for cluster_id in selected_clusters:
            cluster_dir = stage_dir / f"cluster_{cluster_id:02d}"
            if not cluster_dir.exists():
                LOGGER.warning("Cluster directory missing: %s", cluster_dir)
                continue
            process_cluster(stage, cluster_id, cluster_dir, force=args.force)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
