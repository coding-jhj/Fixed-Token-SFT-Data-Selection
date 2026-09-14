"""Assemble the auditable T10 reproduction package without raw BBH rows."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "T10_post_training_reproduction_package_2026-09-13.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(stage: Path, source: Path, target: str) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination = stage / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="t10_reproduction_", dir=ROOT / "work"))
    try:
        copy_file(stage, ROOT / "reproduction/README.md", "README.md")
        copy_file(stage, ROOT / "reproduction/licenses_and_provenance.md", "licenses_and_provenance.md")
        copy_file(stage, ROOT / "reproduction/environment.lock", "environment.lock")
        copy_file(stage, ROOT / "reproduction/CHANGELOG.md", "CHANGELOG.md")
        copy_file(stage, ROOT / "paper/manuscript.md", "paper/manuscript.md")
        copy_file(stage, ROOT / "paper/paper.pdf", "paper/paper.pdf")
        copy_file(stage, ROOT / "paper/ai_advice_and_judgement.md", "paper/ai_advice_and_judgement.md")
        copy_file(stage, ROOT / "scripts/build_paper_pdf.py", "scripts/build_paper_pdf.py")
        copy_file(stage, ROOT / "scripts/reproduce_analysis.ps1", "scripts/reproduce_analysis.ps1")
        for filename in (
            "validate_followup_artifacts.py",
            "analyze_seed_robustness.py",
            "prepare_human_audit.py",
        ):
            copy_file(stage, ROOT / "scripts" / filename, f"scripts/{filename}")
        for filename in ("analyze_final_results.py", "analyze_followup_results.py"):
            copy_file(stage, ROOT / "src" / filename, f"src/{filename}")
        for filename in (
            "select_data.py",
            "contamination_check.py",
            "benchmark_contamination.py",
            "make_eval_subset.py",
            "train_sft.py",
            "evaluate_frozen.py",
        ):
            copy_file(stage, ROOT / "src" / filename, f"src/{filename}")
        copy_file(stage, ROOT / "configs/selection_config.json", "configs/selection_config.json")
        copy_file(stage, ROOT / "work/data_audit.md", "data/source_audit.md")
        copy_file(stage, ROOT / "work/evaluation_subsets_final/manifest.json", "data/evaluation_subset_manifest.json")
        copy_file(stage, ROOT / "work/evaluation_subsets_expanded/manifest.json", "data/evaluation_subset_manifest_expanded.json")
        copy_file(stage, ROOT / "work/selection_manifests/benchmark_contamination_report_expanded.json", "data/benchmark_contamination_report_expanded.json")
        copy_file(stage, ROOT / "work/evaluation_sets/metadata.json", "data/evaluation_sets_metadata.json")

        for path in sorted((ROOT / "work/selection_manifests").glob("manifest_*.csv")):
            copy_file(stage, path, f"data/selection_manifests/{path.name}")
        for path in sorted((ROOT / "work/selection_manifests").glob("manifest_*.summary.json")):
            copy_file(stage, path, f"data/selection_manifests/{path.name}")
        for path in sorted((ROOT / "work/selection_manifests_seed2026").glob("manifest_*.summary.json")):
            copy_file(stage, path, f"data/selection_manifests/{path.name}")
        for path in sorted((ROOT / "work/selection_manifests_seed2026").glob("manifest_*.csv")):
            copy_file(stage, path, f"data/selection_manifests/{path.name}")
        for adapter in sorted((ROOT / "work").glob("main_*_seed*")):
            summary = adapter / "run_summary.json"
            if summary.exists():
                copy_file(stage, summary, f"data/run_summaries/{summary.parent.name}.json")

        for path in sorted((ROOT / "work/results_final").iterdir()):
            if path.is_file():
                copy_file(stage, path, f"results/{path.name}")
        for result_dir in (
            "results_seed2026_long_generation",
            "results_seed_robustness",
            "results_base_long_generation",
            "results_expanded_long_generation",
            "results_reliability",
        ):
            source_dir = ROOT / "work" / result_dir
            if source_dir.exists():
                for path in sorted(source_dir.iterdir()):
                    if path.is_file():
                        copy_file(stage, path, f"results/{result_dir}/{path.name}")
        copy_file(stage, ROOT / "work/human_audit_200/rubric.md", "human_audit/rubric.md")
        copy_file(stage, ROOT / "work/human_audit_200/sampling_manifest.csv", "human_audit/sampling_manifest.csv")

        raw_manifest = []
        for output_dir in (
            "evaluation_final_balanced",
            "evaluation_long_generation",
            "evaluation_seed2026_long_generation",
            "evaluation_base_long_generation",
            "evaluation_expanded_long_generation",
        ):
            root_dir = ROOT / "work" / output_dir
            if not root_dir.exists():
                continue
            for path in sorted(root_dir.rglob("*")):
                if path.is_file():
                    raw_manifest.append(
                        {
                            "workspace_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
        raw_manifest_path = stage / "results/raw_outputs_manifest.json"
        raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8")

        excluded = stage / "results/RAW_OUTPUTS_EXCLUDED.md"
        excluded.write_text(
            "# Raw output redistribution boundary\n\n"
            "Raw evaluation JSONL files remain in the workspace under the primary and follow-up `work/evaluation_*` directories and are represented here only by size and SHA-256 metadata. They are not copied into this ZIP because the local audit did not establish a clear redistribution license for the BBH conversion source. Human audit blind text is also kept in the workspace; the ZIP contains only its rubric and sampling manifest.\n",
            encoding="utf-8",
        )

        package_manifest = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                package_manifest.append(
                    {
                        "path": str(path.relative_to(stage)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        (stage / "PACKAGE_MANIFEST.json").write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
        print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
        print(f"files={len(package_manifest) + 1}")
        print(f"raw_outputs_indexed={len(raw_manifest)}")
    finally:
        shutil.rmtree(stage)


if __name__ == "__main__":
    main()
