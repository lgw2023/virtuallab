from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from virtuallab.api import VirtualLabApp
from virtuallab.graph.model import EdgeType, NodeType


def _load_case_config(path: Path) -> Dict[str, Any]:
    """Load ``case_config.yaml`` without requiring optional PyYAML."""

    try:  # pragma: no cover - use PyYAML when available locally
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(path.read_text())
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise TypeError("case_config.yaml must contain a mapping at the top level")
    return data  # type: ignore[return-value]


def _parse_simple_yaml(raw_text: str) -> Dict[str, Any]:
    """Fallback YAML parser for the limited structure used in the case file."""

    result: Dict[str, Any] = {}
    current_map: Dict[str, str] | None = None
    for raw_line in raw_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  ") and current_map is not None:
            key, value = raw_line.strip().split(":", 1)
            current_map[key.strip()] = value.strip()
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = value
            current_map = None
        else:
            if key == "data_list":
                current_map = {}
                result[key] = current_map
            else:
                result[key] = ""
                current_map = None
    if "data_list" not in result or not isinstance(result["data_list"], dict):
        raise ValueError("case_config.yaml is missing a data_list mapping")
    return result


def _resolve_relative(base: Path, relative: str) -> Path:
    return (base / Path(relative)).resolve()


def _is_fastq(path: Path) -> bool:
    suffixes = path.suffixes
    return any(suffix.endswith("fastq") for suffix in suffixes) or path.name.endswith(".fastq.gz")


def _iter_nodes_of_type(app: VirtualLabApp, node_type: NodeType):
    for node_id in app.graph_store.nodes():
        node = app.graph_store.get_node(node_id)
        if node and node.type is node_type:
            yield node


def test_transcriptome_case_creates_alignment_plan():
    repo_root = Path(__file__).resolve().parent.parent
    case_dir = repo_root / "genome"
    config_path = case_dir / "case_config.yaml"
    config = _load_case_config(config_path)
    data_list: Dict[str, str] = config["data_list"]  # type: ignore[assignment]

    app = VirtualLabApp()

    plan_response = app.handle(
        {
            "action": "create_plan",
            "params": {
                "name": "Mouse mm39 RNA-seq alignment",
                "goal": config["goal_description"],
                "owner": "bioinformatics",
                "labels": ["transcriptomics", "rna-seq", "mouse"],
            },
        }
    )
    plan_id = plan_response["result"]["plan_id"]

    group_subtasks: Dict[str, str] = {}
    for group in ("LoGlu", "HiGlu"):
        subtask_response = app.handle(
            {
                "action": "add_subtask",
                "params": {
                    "plan_id": plan_id,
                    "name": f"{group} replicate alignment",
                    "labels": ["rna-seq", group.lower()],
                },
            }
        )
        group_subtasks[group] = subtask_response["result"]["subtask_id"]

    reference_subtask = app.handle(
        {
            "action": "add_subtask",
            "params": {
                "plan_id": plan_id,
                "name": "Prepare reference assets",
                "labels": ["rna-seq", "reference"],
            },
        }
    )["result"]["subtask_id"]

    fastq_entries: list[dict[str, Any]] = []
    reference_entries: Dict[str, dict[str, Any]] = {}
    for relative_path, description in data_list.items():
        resolved = _resolve_relative(case_dir, relative_path)
        entry = {"relative": relative_path, "description": description, "resolved": resolved}
        if _is_fastq(resolved):
            fastq_entries.append(entry)
        else:
            reference_entries[Path(relative_path).name] = entry

    assert {Path(entry["relative"]).name for entry in fastq_entries} == {
        "SRR1374921.fastq.gz",
        "SRR1374922.fastq.gz",
        "SRR1374923.fastq.gz",
        "SRR1374924.fastq.gz",
    }

    reference_data_ids: Dict[str, str] = {}
    for name, entry in reference_entries.items():
        labels = ["reference"]
        if "adapter" in entry["description"].lower():
            labels.append("adapter")
        if not entry["resolved"].exists():
            labels.append("missing")
        data_response = app.handle(
            {
                "action": "add_data",
                "params": {
                    "payload_ref": str(entry["resolved"]),
                    "format": entry["resolved"].suffix.lstrip(".") or "fasta",
                    "source": "case_config",
                    "description": entry["description"],
                    "labels": labels,
                },
            }
        )
        reference_data_ids[name] = data_response["result"]["data_id"]
        link_response = app.handle(
            {
                "action": "link",
                "params": {
                    "source": plan_id,
                    "target": reference_data_ids[name],
                    "type": EdgeType.USES_DATA.value,
                },
            }
        )
        assert link_response["result"]["edge"]["type"] == EdgeType.USES_DATA.value

    assert {"mm39.fa", "TruSeq3-SE.fa", "mm39.ncbiRefSeq.gtf"}.issubset(reference_data_ids.keys())

    annotation_entry = reference_entries.get("mm39.ncbiRefSeq.gtf")
    assert annotation_entry is not None
    annotation_path = annotation_entry["resolved"]

    index_step = app.handle(
        {
            "action": "add_step",
            "params": {
                "subtask_id": reference_subtask,
                "name": "Build STAR index",
                "tool": "rnaseq-index",
                "inputs": {
                    "reference_fasta": str(reference_entries["mm39.fa"]["resolved"]),
                    "annotation_gtf": str(annotation_path),
                    "output_dir": str((case_dir / "indices").resolve()),
                },
                "labels": ["indexing", "reference"],
            },
        }
    )["result"]["step_id"]

    output_dir = _resolve_relative(case_dir, config["output_dir"])
    step_group_map: Dict[str, str] = {}
    for entry in fastq_entries:
        description = entry["description"]
        group = "LoGlu" if "LoGlu" in description else "HiGlu"
        subtask_id = group_subtasks[group]
        step_response = app.handle(
            {
                "action": "add_step",
                "params": {
                    "subtask_id": subtask_id,
                    "name": f"Align {Path(entry['relative']).name}",
                    "tool": "rna-seq-mapping",
                    "inputs": {
                        "reads": str(entry["resolved"]),
                        "adapter_fasta": str(reference_entries["TruSeq3-SE.fa"]["resolved"]),
                        "reference_fasta": str(reference_entries["mm39.fa"]["resolved"]),
                        "annotation_gtf": str(annotation_path),
                        "output_dir": str(output_dir),
                    },
                    "labels": ["alignment", group.lower()],
                },
            }
        )
        step_id = step_response["result"]["step_id"]
        step_group_map[step_id] = group

    plan_node = app.graph_store.get_node(plan_id)
    assert plan_node is not None
    assert plan_node.attributes["goal"] == config["goal_description"]
    assert {"transcriptomics", "rna-seq", "mouse"}.issubset(plan_node.attributes["labels"])

    subtask_nodes = {node.id: node for node in _iter_nodes_of_type(app, NodeType.SUBTASK)}
    assert set(subtask_nodes) == set(group_subtasks.values()) | {reference_subtask}
    for subtask_id, node in subtask_nodes.items():
        assert EdgeType.CONTAINS.value in app.graph_store.graph.get_edge_data(plan_id, subtask_id)
        if subtask_id in group_subtasks.values():
            assert "rna-seq" in node.attributes["labels"]

    step_nodes = {node.id: node for node in _iter_nodes_of_type(app, NodeType.STEP)}
    assert len(step_nodes) == len(fastq_entries) + 1
    assert index_step in step_nodes
    assert step_nodes[index_step].attributes["inputs"]["reference_fasta"] == str(reference_entries["mm39.fa"]["resolved"])

    observed_reads = {Path(node.attributes["inputs"]["reads"]).name for sid, node in step_nodes.items() if sid != index_step}
    expected_reads = {Path(entry["relative"]).name for entry in fastq_entries}
    assert observed_reads == expected_reads

    for step_id, node in step_nodes.items():
        if step_id == index_step:
            continue
        reads_path = Path(node.attributes["inputs"]["reads"])
        assert reads_path.exists()
        assert node.attributes["subtask_id"] == group_subtasks[step_group_map[step_id]]
        assert node.attributes["inputs"]["annotation_gtf"].endswith("mm39.ncbiRefSeq.gtf")
        assert node.attributes["inputs"]["adapter_fasta"].endswith("TruSeq3-SE.fa")
        assert Path(node.attributes["inputs"]["output_dir"]).name == Path(output_dir).name

    data_nodes = list(_iter_nodes_of_type(app, NodeType.DATA))
    assert len(data_nodes) == len(reference_data_ids)
    for data_node in data_nodes:
        labels = data_node.attributes["labels"]
        assert "reference" in labels
        payload_path = Path(data_node.attributes["payload_ref"])
        if "missing" in labels:
            assert not payload_path.exists()

    for data_id in reference_data_ids.values():
        edge_data = app.graph_store.graph.get_edge_data(plan_id, data_id)
        assert EdgeType.USES_DATA.value in edge_data
