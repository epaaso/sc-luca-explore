#!/usr/bin/env python
"""Run dataset-, stage-, cluster-, and contrast-specific marker extraction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nb_DE_wilcox.modal_DE import MARKER_CONTRASTS, _get_de_impl, common_kwargs
from shared.paths import EXTERNAL_DATA_DIR, EXTERNAL_MARKER_ROOT, MEMBERSHIP_FILES


DATA_DIR = EXTERNAL_DATA_DIR
DEFAULT_OUTPUT_ROOT = EXTERNAL_MARKER_ROOT
PRODUCTION_CONTRASTS = ("normal-vs-normal", "tumor-vs-all")
CLUSTERS = (0, 1, 2)

SURGERIES = [
    {"ext_name": "Zuani_2024_NSCLC", "name": "Zuani", "time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "stage", "pred_name": "Zuani", "obs_unique": False, "obs_has_name": False},
    {"ext_name": "Zuani_2024_NSCLC", "name": "Zuani", "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "stage", "pred_name": "Zuani", "obs_unique": False, "obs_has_name": False},
    {"ext_name": "Deng_Liu_LUAD_2024", "name": "Deng", "time": "I-II", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage", "pred_name": "Deng", "obs_has_name": False},
    {"ext_name": "Deng_Liu_LUAD_2024", "name": "Deng", "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Pathological stage", "pred_name": "Deng", "obs_has_name": False},
    {"ext_name": "Hu_Zhang_2023_NSCLC", "name": "Hu", "time": "III-IV", "cell_key": "cell_type_adjusted", "stage_key": "Clinical Stage", "pred_name": "Hu", "obs_has_name": False},
    {"ext_name": "Trinks_Bishoff_2021_NSCLC", "name": "Bishoff", "time": "III-IV", "cell_key": "cell_type_adjusted", "skip_stages": True, "pred_name": "Bishoff", "obs_has_name": False},
]

ALLOWED_ATLAS = {
    "I-II": {
        "Chen_Zhang_2020", "Lambrechts_Thienpont_2018_6653",
        "Laughney_Massague_2020", "Maynard_Bivona_2020", "Zilionis_Klein_2019",
        "UKIM-V-2", "UKIM-V", "Kim_Lee_2020", "He_Fan_2021",
        "Lambrechts_Thienpont_2018_6149v2", "Lambrechts_Thienpont_2018_6149v1",
    },
    "III-IV": {
        "UKIM-V-2", "Maynard_Bivona_2020", "Zilionis_Klein_2019",
        "Lambrechts_Thienpont_2018_6149v2", "Laughney_Massague_2020",
        "Kim_Lee_2020", "Chen_Zhang_2020", "Lambrechts_Thienpont_2018_6653",
    },
}


def artifact_path(output_root: Path, contrast: str, time: str, dataset: str, cluster: int) -> Path:
    return output_root / contrast / f"{time}_{dataset}_cluster_{cluster}_auc.npy"


def manifest_path(output_root: Path, contrast: str, time: str, dataset: str, cluster: int) -> Path:
    return output_root / contrast / f"{time}_{dataset}_cluster_{cluster}_manifest.json"


def configure_run(spec: dict, contrast: str, output_root: Path, cluster: int) -> dict:
    kwargs = {**common_kwargs, **spec}
    kwargs.update(
        contrast=contrast,
        w_folder=str(output_root),
        cluster_id=cluster,
        membership_csv=str(MEMBERSHIP_FILES[spec["time"]]),
        log_layer="do_log1p",
        load_pair=True,
        skip_visualization=True,
    )
    if contrast == "normal-vs-normal":
        kwargs.update(parallel_pair=True, n_jobs_inner=1, num_processes=4, max_cells_per_type=2000)
    return kwargs


def discover_specs(atlas_only: bool = False) -> list[dict]:
    specs = [] if atlas_only else [dict(spec) for spec in SURGERIES]
    surgery_names = {spec["ext_name"] for spec in SURGERIES}
    atlas_datasets = sorted(
        path.name.removeprefix("filtered_").removesuffix(".h5ad")
        for path in DATA_DIR.glob("filtered_*.h5ad")
        if path.name.removeprefix("filtered_").removesuffix(".h5ad") not in surgery_names
    )
    for time, allowed in ALLOWED_ATLAS.items():
        for dataset in atlas_datasets:
            if dataset not in allowed:
                continue
            specs.append({
                "ext_name": dataset,
                "name": "-".join(dataset.split("_")[0:4:3]),
                "pred_name": "Atlas",
                "time": time,
                "cell_key": "cell_type_adjusted",
                "stage_key": "uicc_stage",
                "obs_has_name": False,
                "gene_feature": "feature_name",
            })
    return specs


def run_markers(
    output_root: Path,
    contrasts: list[str],
    atlas_only: bool = False,
    dry_run: bool = False,
) -> None:
    scheduled = 0
    for spec in discover_specs(atlas_only):
        for cluster in CLUSTERS:
            for contrast in contrasts:
                output = artifact_path(output_root, contrast, spec["time"], spec["ext_name"], cluster)
                if output.exists():
                    print(f"SKIP complete: {output}")
                    continue
                scheduled += 1
                print(
                    f"{'DISCOVER' if dry_run else 'RUN'}: {spec['ext_name']} "
                    f"stage={spec['time']} cluster={cluster} contrast={contrast}"
                )
                if dry_run:
                    continue
                try:
                    _get_de_impl(**configure_run(spec, contrast, output_root, cluster))
                    gc.collect()
                except Exception as error:
                    print(f"ERROR: {spec['ext_name']} stage={spec['time']} cluster={cluster}: {error}")
                    manifest = manifest_path(output_root, contrast, spec["time"], spec["ext_name"], cluster)
                    manifest.parent.mkdir(parents=True, exist_ok=True)
                    manifest.write_text(json.dumps({
                        "dataset": spec["ext_name"],
                        "stage": spec["time"],
                        "cluster": cluster,
                        "contrast": contrast,
                        "status": "failed",
                        "error": str(error),
                    }, indent=2))
    print(f"Scheduled missing runs: {scheduled}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-only", action="store_true", help="Skip surgery datasets.")
    parser.add_argument(
        "--contrast",
        choices=sorted(MARKER_CONTRASTS),
        action="append",
        help="Explicit marker contrast; defaults to normal-vs-normal and tumor-vs-all.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Discover work without writing files.")
    args = parser.parse_args()
    run_markers(args.output_root, args.contrast or list(PRODUCTION_CONTRASTS), args.atlas_only, args.dry_run)


if __name__ == "__main__":
    main()
