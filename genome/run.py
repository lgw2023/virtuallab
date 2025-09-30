"""RNA-seq alignment planning workflow using VirtualLab.

This script demonstrates how to build a transcriptomics analysis plan for the
example data shipped in ``genome/data``.  The pipeline captures the key stages
of a typical single-end RNA-seq experiment:

* register the reference genome, annotation and adapter resources
* build a STAR genome index
* per-replicate quality control, adapter trimming, alignment and gene counting

The resulting execution plan is stored inside the in-memory VirtualLab graph
and summarised to stdout so it can serve as a blueprint for downstream runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

# Ensure the repository root (which contains the ``virtuallab`` package) is
# available on ``sys.path`` when executing the script directly.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from virtuallab.api import VirtualLabApp
from virtuallab.graph.model import EdgeType


CaseConfig = Dict[str, Any]


@dataclass(frozen=True)
class DataEntry:
    """Information about a data asset defined in the case configuration."""

    relative_path: str
    description: str
    resolved: Path

    @property
    def name(self) -> str:
        return Path(self.relative_path).name

    def exists(self) -> bool:
        return self.resolved.exists()

    def is_fastq(self) -> bool:
        suffixes = self.resolved.suffixes
        return any(suffix.endswith("fastq") for suffix in suffixes) or self.name.endswith(".fastq.gz")


@dataclass(frozen=True)
class SampleEntry:
    """Convenience container for FASTQ-driven sample steps."""

    data: DataEntry
    group: str

    @property
    def sample_id(self) -> str:
        # ``SRR1374921.fastq.gz`` -> ``SRR1374921``
        return self.data.name.split(".")[0]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_case_config(path: Path) -> CaseConfig:
    """Load ``case_config.yaml`` with an optional lightweight YAML fallback."""

    try:  # pragma: no cover - prefer PyYAML when present
        import yaml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - executed in lean envs
        return _parse_simple_yaml(path.read_text())
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, MutableMapping):
        raise TypeError("case_config.yaml must contain a mapping at the top level")
    return dict(data)


def _parse_simple_yaml(raw_text: str) -> CaseConfig:
    """Parse the minimal YAML structure shipped with the case study."""

    result: Dict[str, Any] = {}
    current_map: Dict[str, str] | None = None
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.startswith("  ") and current_map is not None:
            key, value = stripped.split(":", 1)
            current_map[key.strip()] = value.strip()
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = value
            current_map = None
        else:
            current_map = {}
            result[key] = current_map
    data_list = result.get("data_list")
    if not isinstance(data_list, Mapping):
        raise ValueError("case_config.yaml is missing a data_list mapping")
    return result


# ---------------------------------------------------------------------------
# VirtualLab orchestration helpers
# ---------------------------------------------------------------------------

def create_plan(app: VirtualLabApp, config: CaseConfig) -> str:
    response = app.handle(
        {
            "action": "create_plan",
            "params": {
                "name": "Mouse mm39 RNA-seq alignment",
                "goal": config.get("goal_description", ""),
                "owner": "bioinformatics",
                "labels": ["transcriptomics", "rna-seq", "mouse"],
            },
        }
    )
    return response["result"]["plan_id"]


def register_reference_data(app: VirtualLabApp, plan_id: str, entries: Iterable[DataEntry]) -> Dict[str, str]:
    """Register non-FASTQ reference assets and link them to the plan."""

    reference_ids: Dict[str, str] = {}
    for entry in entries:
        if entry.is_fastq():
            continue
        labels = ["reference"]
        if "adapter" in entry.description.lower():
            labels.append("adapter")
        if not entry.exists():
            labels.append("missing")
        data_response = app.handle(
            {
                "action": "add_data",
                "params": {
                    "payload_ref": str(entry.resolved),
                    "format": entry.resolved.suffix.lstrip(".") or "fasta",
                    "source": "case_config",
                    "description": entry.description,
                    "labels": labels,
                },
            }
        )
        data_id = data_response["result"]["data_id"]
        reference_ids[entry.name] = data_id
        app.handle(
            {
                "action": "link",
                "params": {
                    "source": plan_id,
                    "target": data_id,
                    "type": EdgeType.USES_DATA.value,
                },
            }
        )
    return reference_ids


def ensure_subtasks(app: VirtualLabApp, plan_id: str) -> Dict[str, str]:
    """Create subtasks for reference prep and experimental groups."""

    subtasks: Dict[str, str] = {}
    for name, labels in (
        ("Prepare reference assets", ["rna-seq", "reference"]),
        ("LoGlu replicate alignment", ["rna-seq", "loglu"]),
        ("HiGlu replicate alignment", ["rna-seq", "higlu"]),
    ):
        response = app.handle(
            {
                "action": "add_subtask",
                "params": {
                    "plan_id": plan_id,
                    "name": name,
                    "labels": labels,
                },
            }
        )
        subtasks[name] = response["result"]["subtask_id"]
    return subtasks


def add_step(app: VirtualLabApp, **params: Any) -> str:
    response = app.handle({"action": "add_step", "params": params})
    return response["result"]["step_id"]


def link_steps(app: VirtualLabApp, source: str, target: str, edge_type: EdgeType = EdgeType.FOLLOWS) -> None:
    app.handle(
        {
            "action": "link",
            "params": {
                "source": source,
                "target": target,
                "type": edge_type.value,
            },
        }
    )


def build_reference_index(
    app: VirtualLabApp,
    subtask_id: str,
    reference_assets: Dict[str, DataEntry],
    output_dir: Path,
) -> str:
    reference_fasta = reference_assets["mm39.fa"].resolved
    annotation = reference_assets.get("mm39.ncbiRefSeq.gtf")
    index_dir = output_dir / "indices" / "star_mm39"
    index_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "reference_fasta": str(reference_fasta),
        "annotation_gtf": str(annotation.resolved if annotation else ""),
        "output_dir": str(index_dir),
        "sjdb_overhang": 100,
    }
    return add_step(
        app,
        subtask_id=subtask_id,
        name="Build STAR index",
        tool="rnaseq-index",
        inputs=inputs,
        labels=["indexing", "reference"],
    )


def plan_sample_workflow(
    app: VirtualLabApp,
    sample: SampleEntry,
    subtask_id: str,
    adapter: DataEntry,
    reference_fasta: DataEntry,
    annotation: DataEntry | None,
    index_step: str,
    output_dir: Path,
) -> List[str]:
    """Create QC, trimming, alignment and quantification steps for one sample."""

    sample_dir = output_dir / sample.group / sample.sample_id
    qc_dir = sample_dir / "qc"
    trim_dir = sample_dir / "trimmed"
    align_dir = sample_dir / "alignment"
    quant_dir = sample_dir / "counts"
    for path in (qc_dir, trim_dir, align_dir, quant_dir):
        path.mkdir(parents=True, exist_ok=True)

    qc_step = add_step(
        app,
        subtask_id=subtask_id,
        name=f"QC {sample.data.name}",
        tool="fastqc",
        inputs={"reads": str(sample.data.resolved), "output_dir": str(qc_dir)},
        labels=["qc", sample.group.lower()],
    )

    trimmed_fastq = trim_dir / f"{sample.sample_id}.trimmed.fastq.gz"
    trim_step = add_step(
        app,
        subtask_id=subtask_id,
        name=f"Trim adapters {sample.data.name}",
        tool="trimming",
        inputs={
            "reads": str(sample.data.resolved),
            "adapter_fasta": str(adapter.resolved),
            "output_fastq": str(trimmed_fastq),
        },
        labels=["trimming", sample.group.lower()],
    )
    link_steps(app, qc_step, trim_step)

    bam_path = align_dir / f"{sample.sample_id}.Aligned.sortedByCoord.out.bam"
    align_step = add_step(
        app,
        subtask_id=subtask_id,
        name=f"Align {sample.data.name}",
        tool="rna-seq-mapping",
        inputs={
            "reads": str(trimmed_fastq),
            "reference_fasta": str(reference_fasta.resolved),
            "annotation_gtf": str(annotation.resolved if annotation else ""),
            "output_bam": str(bam_path),
            "aligner": "STAR",
        },
        labels=["alignment", sample.group.lower()],
    )
    link_steps(app, trim_step, align_step)
    link_steps(app, index_step, align_step, edge_type=EdgeType.DEPENDS_ON)

    counts_file = quant_dir / f"{sample.sample_id}.featureCounts.txt"
    quant_step = add_step(
        app,
        subtask_id=subtask_id,
        name=f"Quantify {sample.data.name}",
        tool="feature-counts",
        inputs={
            "bam": str(bam_path),
            "annotation_gtf": str(annotation.resolved if annotation else ""),
            "output_table": str(counts_file),
        },
        labels=["quantification", sample.group.lower()],
    )
    link_steps(app, align_step, quant_step)
    return [qc_step, trim_step, align_step, quant_step]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    case_dir = Path(__file__).resolve().parent
    config = load_case_config(case_dir / "case_config.yaml")

    data_entries = [
        DataEntry(relative_path=relative, description=desc, resolved=(case_dir / relative).resolve())
        for relative, desc in config["data_list"].items()
    ]

    app = VirtualLabApp()
    plan_id = create_plan(app, config)
    subtasks = ensure_subtasks(app, plan_id)

    reference_entries = {entry.name: entry for entry in data_entries if not entry.is_fastq()}
    reference_ids = register_reference_data(app, plan_id, reference_entries.values())

    output_dir = (case_dir / config.get("output_dir", "output")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_subtask_id = subtasks["Prepare reference assets"]
    index_step = build_reference_index(app, reference_subtask_id, reference_entries, output_dir)

    adapter_entry = reference_entries["TruSeq3-SE.fa"]
    reference_fasta_entry = reference_entries["mm39.fa"]
    annotation_entry = reference_entries.get("mm39.ncbiRefSeq.gtf")

    samples_by_group: Dict[str, List[SampleEntry]] = defaultdict(list)
    for entry in data_entries:
        if not entry.is_fastq():
            continue
        group = "LoGlu" if "LoGlu" in entry.description else "HiGlu"
        samples_by_group[group].append(SampleEntry(entry, group))

    created_steps: Dict[str, List[str]] = {}
    for group, samples in samples_by_group.items():
        subtask_id = subtasks[f"{group} replicate alignment"]
        for sample in samples:
            created_steps[sample.sample_id] = plan_sample_workflow(
                app,
                sample,
                subtask_id,
                adapter_entry,
                reference_fasta_entry,
                annotation_entry,
                index_step,
                output_dir,
            )

    print("RNA-seq analysis plan successfully created")
    print(f"Plan ID: {plan_id}")
    print(f"Registered reference assets: {sorted(reference_ids)}")
    print("Sample step overview:")
    for sample_id, steps in sorted(created_steps.items()):
        print(f"  - {sample_id}: {len(steps)} steps ({', '.join(steps)})")


if __name__ == "__main__":
    main()
