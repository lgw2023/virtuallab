# virtuallab

VirtualLab 是一个基于 NetworkX 的轻量级实验编排与知识图谱原型，用于模拟从实验规划、步骤拆解到知识沉淀的最小闭环。仓库当前聚焦于搭建清晰的代码骨架和数据结构，以便后续扩展执行能力、自动连边与持久化。

## 快速上手

```python
from virtuallab import VirtualLab_tool

response = VirtualLab_tool({
    "action": "create_plan",
    "params": {"name": "异常检测", "goal": "定位销售异常"}
})

print(response)
# {
#     "ok": True,
#     "result": {"plan_id": "plan_xxx"},
#     "events": [...],
#     "graph_delta": {
#         "added_nodes": [...],
#         "added_edges": [],
#         "updated_nodes": [],
#     },
# }
```

调用统一入口 `VirtualLab_tool(payload)`，内部由 `VirtualLabApp` 负责解析 action、分发处理并返回标准化响应：

1. `VirtualLabApp.handle` 校验 action 并读取参数。
2. `ActionRouter` 将 action 分发至对应处理函数。
3. 处理函数操作 `GraphStore` 并返回 `GraphDelta`，同时由 `EventBus` 记录事件。
4. 响应中始终包含 `ok`、`result`、`events` 与结构化的 `graph_delta`。

## 目录结构

```
virtuallab/
├── __init__.py             # 暴露 VirtualLab_tool
├── api.py                  # VirtualLabApp 核心应用容器与默认动作实现
├── router.py               # 动作路由注册与分发
├── graph/
│   ├── ids.py              # ID 与时间戳工具
│   ├── model.py            # 节点/边枚举、GraphDelta、GraphSchema
│   ├── store.py            # 基于 networkx 的存储封装
│   ├── query.py            # by_type、timeline、neighbors 查询能力
│   └── rules.py            # 自动连边规则占位
├── exec/
│   ├── runner.py           # StepRunner，负责调用执行适配器
│   └── adapters/           # Engineer、Local、OpenAI Model 等适配器骨架
├── knowledge/
│   └── summarize.py        # SummaryService，对外部总结适配器的轻封装
├── obs/
│   └── events.py           # Event、EventBus，可追加查询事件
├── persist/
│   ├── export.py           # GraphExporter，占位支持 JSON/GraphML
│   └── snapshot.py         # SnapshotManager，提供快照与回滚接口
└── llm.py                  # OpenAI 相关客户端与工具（占位）
```

## 默认动作（Actions）

| Action 名称      | 功能概述 | 关键参数 |
|------------------|----------|----------|
| `create_plan`    | 创建 `Plan` 节点并写入基础元数据 | `name`, `goal`, `owner`, `status`（可选） |
| `add_subtask`    | 在 Plan 下新增 `Subtask` 并建立 `CONTAINS` 边 | 必需 `plan_id`，可自定义 `priority` |
| `add_step`       | 为 Subtask 新增 `Step` 节点并建立 `CONTAINS` 边 | 必需 `subtask_id`，包含工具、输入、状态信息 |
| `add_data`       | 创建 `Data` 节点，记录数据引用元信息 | 支持记录来源、格式、引用指针 |
| `link`           | 在任意节点间创建一条有向边 | `source`, `target`, `type`, `attributes` |
| `query`          | 基于 `QueryService` 查询图谱 | `kind` ∈ {`by_type`, `timeline`, `neighbors`} |

以下动作已注册占位符，当前会返回「未实现」的提示：`run_step`、`summarize`、`record_note`、`auto_link`、`export_graph`、`snapshot`、`rollback`。

## 图谱模型与操作

### 节点与边

- 节点类型（`NodeType`）：`Plan`、`Subtask`、`Step`、`Data`，并预留 `Result`、`Note`、`Agent`。
- 边类型（`EdgeType`）：`CONTAINS`、`USES_DATA`、`PRODUCES`、`DERIVES`、`DEPENDS_ON`、`CAUSED_BY`、`FOLLOWS`、`ASSOCIATED_WITH`。
- `GraphSchema` 维护各节点/边默认字段，`coerce_node_payload` 与 `coerce_edge_payload` 提供轻量校验工具。
- `GraphStore` 基于 `networkx.MultiDiGraph`，支持增量 `GraphDelta` 应用与节点、边查询。

### 查询服务

`QueryService` 暴露三类常用模式：

- `by_type(node_type, **filters)`：按类型与属性过滤节点。
- `timeline(scope, include)`：结合 `created_at`/`executed_at` 字段排序，支持按 Plan 或节点类型过滤。
- `neighbors(node_id, hop=1, edge_types=None)`：遍历指定跳数内邻居，并可限定边类型。

### GraphDelta 序列化

`VirtualLabApp` 会将 `GraphDelta` 序列化为三个数组：`added_nodes`、`added_edges`、`updated_nodes`。若无变更则返回空数组，便于外部系统回放。

## 执行、总结与持久化扩展点

- `StepRunner` 维护执行适配器注册表，`run(tool, step_id, payload)` 会分发到具体适配器，实现按工具类型的灵活调度。
- `SummaryService` 抽象总结能力，调用外部适配器产出结构化结果。
- `GraphExporter` 与 `SnapshotManager` 分别负责图谱导出与快照回滚，当前实现为占位，可在后续迭代中接入持久化。

## 事件与可观测性

`EventBus` 使用 append-only 列表存储事件。每次 action 执行后都会写入包含时间戳、级别、消息与上下文的事件，可通过 `history()` 获取完整时间线，便于调试与追踪。

## 后续规划

短期方向包括：

1. 打通 `run_step`、`summarize` 等动作，将执行器与总结服务纳入主流程。
2. 实现自动连边规则（时序、依赖、因果等）。
3. 完善导出、快照回滚与外部持久化集成。
4. 增加端到端示例与测试用例，覆盖典型交互路径。

当前代码骨架为后续迭代预留了清晰的模块边界，可按需替换或扩展各层能力。
