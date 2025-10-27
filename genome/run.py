"""RNA-seq alignment planning workflow using VirtualLab.

This script demonstrates how to build a transcriptomics analysis plan for the
example data shipped in ``genome/data``.  The pipeline captures the key stages
of a typical single-end RNA-seq experiment:

* register the reference genome, annotation and adapter resources
* build a STAR genome index
* per-replicate quality control, adapter trimming, alignment and gene counting
* aggregate counts into a matrix and plan a differential expression comparison

The resulting execution plan is stored inside the in-memory VirtualLab graph
and summarised to stdout so it can serve as a blueprint for downstream runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

# Ensure the repository root (which contains the ``virtuallab`` package) is
# available on ``sys.path`` when executing the script directly.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from virtuallab.api import VirtualLabApp
from virtuallab.graph.model import EdgeType, NodeSpec, NodeType


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


@dataclass(frozen=True)
class SampleWorkflowResult:
    """Summary of all steps constructed for one sequencing replicate."""

    sample: SampleEntry
    steps: List[str]
    terminal_step: str
    counts_file: Path


@dataclass(frozen=True)
class StageResult:
    """Pass planning results between sequential workflow stages."""

    anchor_step: str
    steps: Mapping[str, str]
    artifacts: Mapping[str, Any]
    order: List[str] | None = None

    def primary_step(self) -> str:
        return self.steps.get("primary", self.anchor_step)

    def ordered_steps(self) -> List[str]:
        if self.order:
            return list(self.order)
        if self.steps:
            return list(self.steps.values())
        if self.anchor_step:
            return [self.anchor_step]
        return []


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
        ("Differential expression analysis", ["rna-seq", "differential-expression"]),
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
    return response["result"]["step_id"], response


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
) -> StageResult:
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
    index_step, index_response = add_step(
        app,
        subtask_id=subtask_id,
        name="Build STAR index",
        tool="rnaseq-index",
        inputs=inputs,
        labels=["indexing", "reference"],
    )
    return StageResult(
        anchor_step=index_step,
        steps={"primary": index_step},
        artifacts={"primary": index_dir, "raw_response": index_response},
        order=[index_step],
    )


def plan_sample_workflow(
    app: VirtualLabApp,
    sample: SampleEntry,
    subtask_id: str,
    adapter: DataEntry,
    reference_fasta: DataEntry,
    annotation: DataEntry | None,
    upstream_stage: StageResult,
    output_dir: Path,
) -> SampleWorkflowResult:
    """Create per-sample steps while depending on a prior planning stage."""

    sample_dir = output_dir / sample.group / sample.sample_id
    qc_dir = sample_dir / "qc"
    trim_dir = sample_dir / "trimmed"
    align_dir = sample_dir / "alignment"
    quant_dir = sample_dir / "counts"
    for path in (qc_dir, trim_dir, align_dir, quant_dir):
        path.mkdir(parents=True, exist_ok=True)

    qc_step, qc_response = add_step(
        app,
        subtask_id=subtask_id,
        name=f"QC {sample.data.name}",
        tool="fastqc",
        inputs={"reads": str(sample.data.resolved), "output_dir": str(qc_dir)},
        labels=["qc", sample.group.lower()],
    )

    trimmed_fastq = trim_dir / f"{sample.sample_id}.trimmed.fastq.gz"
    trim_step, trim_response = add_step(
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
    align_step, align_response = add_step(
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
    link_steps(app, upstream_stage.primary_step(), align_step, edge_type=EdgeType.DEPENDS_ON)

    counts_file = quant_dir / f"{sample.sample_id}.featureCounts.txt"
    quant_step, quant_response = add_step(
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
    return SampleWorkflowResult(
        sample=sample,
        steps=[qc_step, trim_step, align_step, quant_step],
        terminal_step=quant_step,
        counts_file=counts_file,
    )


def plan_differential_expression(
    app: VirtualLabApp,
    subtask_id: str,
    upstream_samples: Iterable[SampleWorkflowResult],
    output_dir: Path,
) -> StageResult:
    """Add post-sample steps that consume upstream sample workflows."""

    results = list(upstream_samples)
    if not results:
        return StageResult(anchor_step="", steps={}, artifacts={})

    de_dir = output_dir / "differential_expression"
    de_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = de_dir / "counts_matrix.tsv"
    de_table_path = de_dir / "differential_expression.tsv"

    counts_inputs = {
        "counts_files": {res.sample.sample_id: str(res.counts_file) for res in results},
        "output_matrix": str(matrix_path),
    }
    counts_step, counts_response = add_step(
        app,
        subtask_id=subtask_id,
        name="Assemble counts matrix",
        tool="counts-matrix",
        inputs=counts_inputs,
        labels=["quantification", "aggregation"],
    )

    for res in results:
        link_steps(app, res.terminal_step, counts_step, edge_type=EdgeType.DEPENDS_ON)

    sample_groups = {res.sample.sample_id: res.sample.group for res in results}
    de_inputs = {
        "counts_matrix": str(matrix_path),
        "sample_groups": sample_groups,
        "test_design": {"control": "LoGlu", "treatment": "HiGlu"},
        "output_table": str(de_table_path),
    }
    de_step, de_response = add_step(
        app,
        subtask_id=subtask_id,
        name="Differential expression analysis",
        tool="deseq2",
        inputs=de_inputs,
        labels=["differential-expression"],
    )
    link_steps(app, counts_step, de_step)

    return StageResult(
        anchor_step=de_step,
        steps={"aggregation": counts_step, "primary": de_step},
        artifacts={
            "primary": de_table_path,
            "counts_matrix": matrix_path,
        },
        order=[counts_step, de_step],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_execution_prompt(step: NodeSpec) -> str:
    """Create a textual prompt describing how to execute ``step``."""

    name = step.attributes.get("name", "")
    tool = step.attributes.get("tool", "")
    inputs = step.attributes.get("inputs", {})
    labels = ", ".join(step.attributes.get("labels", []))
    instructions = [
        "You are the execution agent for a VirtualLab workflow step.",
        f"Step ID: {step.id}",
        f"Step name: {name}",
        f"Suggested tool: {tool}",
        f"Labels: {labels}" if labels else "Labels: (none)",
        "Inputs (JSON):",
        json.dumps(inputs, indent=2, sort_keys=True),
        "",
        "Goals:",
        "1. Perform the bioinformatics operation that corresponds to the step.",
        "2. Operate directly on the provided input files or paths.",
        "3. Ensure the expected output artefacts are created at the paths listed in the inputs.",
        "4. Use shell commands (via the bash tool) or lightweight Python scripts when external tools are missing.",
        "5. At the end, summarise the actions taken and key results produced.",
    ]
    return "\n".join(instructions)


def _execute_steps(app: VirtualLabApp, step_ids: Iterable[str]) -> None:
    """Execute each step in ``step_ids`` sequentially using the Engineer runner."""

    for step_id in step_ids:
        step = app.graph_store.get_node(step_id)
        if step is None or step.type is not NodeType.STEP:
            print(f"Skipping execution for unknown step: {step_id}")
            continue
        prompt = _build_execution_prompt(step)
        print(f"Executing step {step_id} ({step.attributes.get('name', '')}) using Engineer runner")
        payload = {"text": prompt, "tools": ["shell_bash"]}
        try:
            response = app.handle(
                {
                    "action": "run_step",
                    "params": {"step_id": step_id, "tool": "engineer", "payload": payload},
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Execution failed for step {step_id}: {exc}")
            continue
        result = response.get("result", {})
        status = result.get("status", "unknown")
        print(f"Step {step_id} execution status: {status}")
        if result.get("output"):
            print(f"Step {step_id} output: {result['output']}")


def main() -> None:
    case_dir = Path(__file__).resolve().parent
    config = load_case_config(case_dir / "case_config.yaml")

    data_entries = [
        DataEntry(relative_path=relative, description=desc, resolved=(case_dir / relative).resolve())
        for relative, desc in config["data_list"].items()
    ]

    app = VirtualLabApp()
    plan_id = create_plan(app, config)
    print(f"plan_id: {plan_id}")
    subtasks = ensure_subtasks(app, plan_id)
    print(f"subtasks: {subtasks}")

    reference_entries = {entry.name: entry for entry in data_entries if not entry.is_fastq()}
    print(f"reference_entries: {reference_entries}")
    reference_ids = register_reference_data(app, plan_id, reference_entries.values())
    print(f"reference_ids: {reference_ids}")

    output_dir = (case_dir / config.get("output_dir", "output")).resolve()
    print(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_subtask_id = subtasks["Prepare reference assets"]
    print(f"reference_subtask_id: {reference_subtask_id}")
    reference_stage = build_reference_index(app, reference_subtask_id, reference_entries, output_dir)
    print(f"reference_stage: {reference_stage}")

    adapter_entry = reference_entries["TruSeq3-SE.fa"]
    print(f"adapter_entry: {adapter_entry}")
    reference_fasta_entry = reference_entries["mm39.fa"]
    print(f"reference_fasta_entry: {reference_fasta_entry}")
    annotation_entry = reference_entries.get("mm39.ncbiRefSeq.gtf")
    print(f"annotation_entry: {annotation_entry}")

    samples_by_group: Dict[str, List[SampleEntry]] = defaultdict(list)
    for entry in data_entries:
        if not entry.is_fastq():
            continue
        group = "LoGlu" if "LoGlu" in entry.description else "HiGlu"
        samples_by_group[group].append(SampleEntry(entry, group))
    print(f"samples_by_group: {samples_by_group}")

    created_steps: Dict[str, List[str]] = {}
    sample_results: List[SampleWorkflowResult] = []
    for group, samples in samples_by_group.items():
        subtask_id = subtasks[f"{group} replicate alignment"]
        for sample in samples:
            result = plan_sample_workflow(
                app,
                sample,
                subtask_id,
                adapter_entry,
                reference_fasta_entry,
                annotation_entry,
                reference_stage,
                output_dir,
            )
            created_steps[sample.sample_id] = result.steps
            sample_results.append(result)
    print(f"created_steps: {created_steps}")

    de_subtask_id = subtasks["Differential expression analysis"]
    de_stage = plan_differential_expression(app, de_subtask_id, sample_results, output_dir)
    print(f"de_stage: {de_stage}")

    execution_order = reference_stage.ordered_steps()
    execution_order.extend(step_id for res in sample_results for step_id in res.steps)
    execution_order.extend(de_stage.ordered_steps())

    print("RNA-seq analysis plan successfully created")
    print(f"Plan ID: {plan_id}")
    print(f"Registered reference assets: {sorted(reference_ids)}")
    print("Sample step overview:")
    for sample_id, steps in sorted(created_steps.items()):
        print(f"  - {sample_id}: {len(steps)} steps ({', '.join(steps)})")
    if de_stage:
        print("Differential expression steps:")
        for name, step_id in de_stage.steps.items():
            print(f"  - {name}: {step_id}")

    # Convert NodeDataView/EdgeDataView to list before json serialization
    print(f"app.graph_store.graph nodes: {json.dumps(list(app.graph_store.graph.nodes(data=True)), indent=2)}")

    if execution_order:
        print("\nStarting execution of planned steps using Engineer...")
        _execute_steps(app, execution_order)
    # print(f"app.graph_store.graph edges: {json.dumps(list(app.graph_store.graph.edges(data=True)), indent=2)}")


if __name__ == "__main__":
    main()
