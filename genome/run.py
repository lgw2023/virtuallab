"""RNA-seq alignment planning workflow using the VirtualLab handles API."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sys
from xmlrpc.client import Fault

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from virtuallab.api import PlanHandle, StepHandle, SubtaskHandle, VirtualLabApp
from virtuallab.graph.model import EdgeType, NodeSpec, NodeType


CaseConfig = Mapping[str, object]


@dataclass(frozen=True)
class DataAsset:
    """Representation of one entry in ``case_config.yaml``."""

    source: str
    description: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def format(self) -> str:
        suffix = self.path.suffix.lstrip(".")
        return suffix or "unknown"

    def exists(self) -> bool:
        return self.path.exists()

    def is_fastq(self) -> bool:
        suffixes = self.path.suffixes
        return any(part.endswith("fastq") for part in suffixes) or self.name.endswith(".fastq.gz")

    def sample_id(self) -> str:
        return self.name.split(".")[0]

    def experimental_group(self) -> str:
        lower_desc = self.description.lower()
        if "loglu" in lower_desc:
            return "LoGlu"
        if "higlu" in lower_desc:
            return "HiGlu"
        return "Unknown"


@dataclass(frozen=True)
class RegisteredAsset:
    asset: DataAsset
    data_id: str


@dataclass
class ReferenceStage:
    index_step: StepHandle
    index_dir: Path


@dataclass
class SamplePlan:
    sample: DataAsset
    group: str
    steps: list[StepHandle]
    counts_table: Path

    @property
    def terminal_step(self) -> StepHandle:
        return self.steps[-1]


@dataclass
class DifferentialExpressionPlan:
    counts_step: StepHandle
    differential_step: StepHandle
    matrix_path: Path
    results_path: Path


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_case_config(path: Path) -> CaseConfig:
    """Load ``case_config.yaml`` with an optional lightweight YAML fallback."""

    raw_text = path.read_text()
    try:  # pragma: no cover - prefer PyYAML when available
        import yaml  # type: ignore

        data = yaml.safe_load(raw_text)
        if not isinstance(data, Mapping):
            raise TypeError("case_config.yaml must contain a mapping at the top level")
        return data
    except ModuleNotFoundError:  # pragma: no cover - executed in lean envs
        return _parse_simple_yaml(raw_text)


def _parse_simple_yaml(raw_text: str) -> CaseConfig:
    """Parse the minimal YAML structure shipped with the case study."""

    result: dict[str, object] = {}
    current_map: dict[str, str] | None = None
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


def load_assets(config: CaseConfig, case_dir: Path) -> list[DataAsset]:
    data_list = config.get("data_list")
    if not isinstance(data_list, Mapping):
        raise TypeError("case_config.yaml must define a data_list mapping")

    assets: list[DataAsset] = []
    for relative, desc in data_list.items():
        resolved = (case_dir / relative).resolve()
        assets.append(DataAsset(source=str(relative), description=str(desc), path=Path(relative)))
    return assets


def create_plan(app: VirtualLabApp, config: CaseConfig) -> PlanHandle:
    goal = str(config.get("goal_description", ""))
    return app.plan(
        name="Mouse mm39 RNA-seq alignment",
        goal=goal,
        owner="bioinformatics",
        labels=["transcriptomics", "rna-seq", "mouse"],
    )


def register_references(plan: PlanHandle, assets: Iterable[DataAsset]) -> dict[str, RegisteredAsset]:
    registered: dict[str, RegisteredAsset] = {}
    for asset in assets:
        labels = ["reference"]
        if "adapter" in asset.description.lower():
            labels.append("adapter")
        if not asset.exists():
            labels.append("missing")
        handle = plan.register_data(
            payload_ref=str(asset.path),
            format=asset.format,
            source="case_config",
            labels=labels,
            attributes={"description": asset.description},
        )
        registered[asset.name] = RegisteredAsset(asset=asset, data_id=handle.id)
    return registered


def create_subtasks(plan: PlanHandle) -> dict[str, SubtaskHandle]:
    return {
        "reference": plan.add_subtask(name="Prepare reference assets", labels=["rna-seq", "reference"]),
        "LoGlu": plan.add_subtask(name="LoGlu replicate alignment", labels=["rna-seq", "loglu"]),
        "HiGlu": plan.add_subtask(name="HiGlu replicate alignment", labels=["rna-seq", "higlu"]),
        "de": plan.add_subtask(
            name="Differential expression analysis",
            labels=["rna-seq", "differential-expression"],
        ),
    }


def plan_reference_stage(
    subtask: SubtaskHandle,
    references: Mapping[str, RegisteredAsset],
    output_dir: Path,
) -> ReferenceStage:
    genome = references["mm39.fa"].asset
    annotation = references.get("mm39.ncbiRefSeq.gtf")

    index_dir = output_dir / "indices" / "HISAT2_mm39"
    index_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "reference_fasta": str(genome.path),
        "annotation_gtf": str(annotation.asset.path) if annotation else "",
        "output_dir": str(index_dir),
    }
    step = subtask.add_step(
        name="Build HISAT2 index",
        tool="HISAT2",
        inputs=inputs,
        labels=["indexing", "reference"],
    )
    return ReferenceStage(index_step=step, index_dir=index_dir)


def plan_sample_pipeline(
    app: VirtualLabApp,
    sample: DataAsset,
    subtask: SubtaskHandle,
    adapter: DataAsset,
    annotation: DataAsset | None,
    reference_stage: ReferenceStage,
    output_dir: Path,
) -> SamplePlan:
    group = sample.experimental_group()
    sample_dir = output_dir / group / sample.sample_id()
    qc_dir = sample_dir / "qc"
    trim_dir = sample_dir / "trimmed"
    align_dir = sample_dir / "alignment"
    quant_dir = sample_dir / "counts"
    for directory in (qc_dir, trim_dir, align_dir, quant_dir):
        directory.mkdir(parents=True, exist_ok=True)

    steps: list[StepHandle] = []

    qc_step = subtask.add_step(
        name=f"QC {sample.name}",
        tool="fastqc",
        inputs={"reads": str(sample.path), "output_dir": str(qc_dir)},
        labels=["qc", group.lower()],
    )
    steps.append(qc_step)

    trimmed_fastq = trim_dir / f"{sample.sample_id()}.trimmed.fastq.gz"
    trim_step = subtask.add_step(
        name=f"Trim adapters {sample.name}",
        tool="cutadapt",
        inputs={
            "reads": str(sample.path),
            "adapter_fasta": str(adapter.path),
            "output_fastq": str(trimmed_fastq),
        },
        labels=["cutadapt", group.lower()],
    )
    steps.append(trim_step)
    app.link(source=qc_step, target=trim_step, type=EdgeType.FOLLOWS)

    bam_path = align_dir / f"{sample.sample_id()}.bam"
    align_step = subtask.add_step(
        name=f"Align {sample.name}",
        tool="hisat2",
        inputs={
            "reads": str(trimmed_fastq),
            "reference_genome_index": "find the related HISAT2 index file under the data or output directory by yourself",
            "annotation_gtf": str(annotation.path) if annotation else "",
            "output_bam": str(bam_path),
            "aligner": "HISAT2",
        },
        labels=["alignment", group.lower()],
    )
    steps.append(align_step)
    app.link(source=trim_step, target=align_step, type=EdgeType.FOLLOWS)
    app.link(source=reference_stage.index_step, target=align_step, type=EdgeType.DEPENDS_ON)

    counts_file = quant_dir / f"{sample.sample_id()}.featureCounts.txt"
    quant_step = subtask.add_step(
        name=f"Quantify {sample.name}",
        tool="featureCounts",
        inputs={
            "bam": str(bam_path),
            "annotation_gtf": str(annotation.path) if annotation else "",
            "output_table": str(counts_file),
        },
        labels=["quantification", group.lower()],
    )
    steps.append(quant_step)
    app.link(source=align_step, target=quant_step, type=EdgeType.FOLLOWS)

    return SamplePlan(sample=sample, group=group, steps=steps, counts_table=counts_file)


def plan_differential_expression(
    app: VirtualLabApp,
    subtask: SubtaskHandle,
    sample_plans: Sequence[SamplePlan],
    output_dir: Path,
) -> DifferentialExpressionPlan | None:
    if not sample_plans:
        return None

    de_dir = output_dir / "differential_expression"
    de_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = de_dir / "counts_matrix.tsv"
    de_table_path = de_dir / "differential_expression.tsv"

    counts_inputs = {
        "counts_files": {plan.sample.sample_id(): str(plan.counts_table) for plan in sample_plans},
        "output_matrix": str(matrix_path),
    }
    counts_step = subtask.add_step(
        name="Assemble counts matrix",
        tool="pandas",
        inputs=counts_inputs,
        labels=["quantification", "aggregation"],
    )
    for plan in sample_plans:
        app.link(source=plan.terminal_step, target=counts_step, type=EdgeType.DEPENDS_ON)

    group_map = {plan.sample.sample_id(): plan.group for plan in sample_plans}
    de_inputs = {
        "counts_matrix": str(matrix_path),
        "sample_groups": group_map,
        "test_design": {"control": "LoGlu", "treatment": "HiGlu"},
        "output_table": str(de_table_path),
    }
    de_step = subtask.add_step(
        name="Differential expression analysis",
        tool="R script: DESeq2",
        inputs=de_inputs,
        labels=["differential-expression"],
    )
    app.link(source=counts_step, target=de_step, type=EdgeType.FOLLOWS)

    return DifferentialExpressionPlan(
        counts_step=counts_step,
        differential_step=de_step,
        matrix_path=matrix_path,
        results_path=de_table_path,
    )


def build_execution_prompt(step: NodeSpec, history: str) -> str:
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
        "6. If the step is related to the previous steps, use the history summary to provide context.",

    ]
    if history:
        instructions.extend(["History result summary:", history,])
    return "\n".join(instructions)


def execute_steps(app: VirtualLabApp, steps: Sequence[StepHandle], history: str) -> str:
    history_summary = "" if not history else history
    result = defaultdict()
    for handle in steps:
        step_node = handle.node()
        prompt = build_execution_prompt(step_node, history_summary)
        print(f"Executing {step_node.id} ({step_node.attributes.get('name', '')})\nprompt: {prompt}")
        payload = {"text": prompt, "tools": ["shell_bash"]}
        try:
            response = app.run_step(step=handle, tool="engineer", payload=payload)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Execution failed for {step_node.id}: {exc}")
            continue
        try:
            print(f"response: {json.dumps(response, ensure_ascii=False)}")
        except:
            print(f"response: {response}")
        result = response.get("result", {})
        status = result.get("status", "unknown")
        assert result.get("brief_output")
        if result.get("brief_output"):
            history_summary += str(result["brief_output"])
        if result.get("output"):
            print(f"  output: {result['output']}")
    return str(result["brief_output"])

def summarise_plan(
    plan: PlanHandle,
    references: Mapping[str, RegisteredAsset],
    sample_plans: Sequence[SamplePlan],
    de_plan: DifferentialExpressionPlan | None,
) -> None:
    print("RNA-seq analysis plan successfully created")
    print(f"Plan ID: {plan.id}")
    print("Registered reference assets:")
    for name, registered in sorted(references.items()):
        status = "available" if registered.asset.exists() else "missing"
        print(f"  - {name}: {registered.data_id} ({status})")

    print("Sample step overview:")
    for plan_entry in sorted(sample_plans, key=lambda item: item.sample.sample_id()):
        step_ids = [step.id for step in plan_entry.steps]
        print(
            f"  - {plan_entry.sample.sample_id()} ({plan_entry.group}): "
            f"{len(step_ids)} steps -> {', '.join(step_ids)}"
        )

    if de_plan:
        print("Differential expression steps:")
        print(f"  - counts matrix: {de_plan.counts_step.id}")
        print(f"  - differential expression: {de_plan.differential_step.id}")

    timeline = plan.timeline(
        include=[NodeType.SUBTASK.value, NodeType.STEP.value, NodeType.DATA.value]
    )
    print("\nPlan timeline excerpt:")
    print(json.dumps(timeline[:10], indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(execute: bool = False) -> None:
    os.chdir('/Users/liguowei/ubuntu/virtuallab/genome')
    case_dir = Path("") # Path(__file__).resolve().parent
    config = load_case_config(case_dir / "case_config.yaml")
    assets = load_assets(config, case_dir)
    print(f"assets: {assets}")

    reference_assets = [asset for asset in assets if not asset.is_fastq()]
    sample_assets = [asset for asset in assets if asset.is_fastq()]
    print(f"reference_assets: {reference_assets}")
    print(f"sample_assets: {sample_assets}")

    app = VirtualLabApp()
    plan = create_plan(app, config)

    references = register_references(plan, reference_assets)
    subtasks = create_subtasks(plan)

    output_dir = case_dir / str(config.get("output_dir", "output")) # (case_dir / str(config.get("output_dir", "output"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_stage = plan_reference_stage(subtasks["reference"], references, output_dir)

    adapter_asset = references.get("TruSeq3-SE.fa")
    if adapter_asset is None:
        raise KeyError("TruSeq3-SE.fa must be defined in case_config.yaml")
    annotation_asset = references.get("mm39.ncbiRefSeq.gtf")

    sample_plans: list[SamplePlan] = []
    for asset in sample_assets:
        group = asset.experimental_group()
        subtask = subtasks.get(group)
        if subtask is None:
            raise KeyError(f"No subtask registered for group '{group}'")
        plan_entry = plan_sample_pipeline(
            app=app,
            sample=asset,
            subtask=subtask,
            adapter=adapter_asset.asset,
            annotation=annotation_asset.asset if annotation_asset else None,
            reference_stage=reference_stage,
            output_dir=output_dir,
        )
        sample_plans.append(plan_entry)

    de_plan = plan_differential_expression(app, subtasks["de"], sample_plans, output_dir)

    summarise_plan(plan, references, sample_plans, de_plan)

    if execute:
        print("\nStarting execution of planned steps using Engineer...")
        history = execute_steps(app, [reference_stage.index_step], "")
        for plan_entry in sample_plans:
            history += execute_steps(app, plan_entry.steps, history)
        if de_plan:
            history += execute_steps(app, [de_plan.counts_step, de_plan.differential_step], history)


if __name__ == "__main__":
    main(execute=True)
