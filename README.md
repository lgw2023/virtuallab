# virtuallab

VirtualLab 是一个基于 NetworkX 的轻量级实验编排与知识图谱原型，用于模拟从实验规划、步骤拆解到知识沉淀的最小闭环。当前仓库聚焦于搭建清晰的代码骨架与数据结构，便于后续逐步扩展执行能力与自动连边策略。

## 核心调用流程

1. 外部调用统一入口 `VirtualLab_tool(payload)`。
2. `VirtualLabApp` 利用 `ActionRouter` 将 action 分发到对应的处理函数。
3. 处理函数通过 `GraphStore` 操作 NetworkX 图，同时向 `EventBus` 追加事件。
4. 返回值包含标准化的 `result`、`events` 以及本次操作的 `graph_delta`。

```python
from virtuallab import VirtualLab_tool

plan = VirtualLab_tool({
    "action": "create_plan",
    "params": {"name": "异常检测", "goal": "定位销售异常"}
})
```

## 代码结构

```
virtuallab/
├── __init__.py             # 暴露 VirtualLab_tool
├── api.py                  # VirtualLabApp 核心应用容器
├── router.py               # 动作路由与注册
├── llm.py                  # OpenAI 相关客户端与工具（待整理）
├── graph/
│   ├── ids.py              # ID 与时间戳工具
│   ├── model.py            # 节点/边定义、GraphDelta
│   ├── store.py            # 基于 networkx 的存储封装
│   ├── query.py            # 结构化查询工具
│   └── rules.py            # 自动连边规则占位
├── exec/
│   ├── runner.py           # StepRunner，负责调用执行适配器
│   └── adapters/
│       ├── engineer.py     # Engineer 智能体适配器骨架
│       ├── local.py        # 本地函数适配器（可注册可调用）
│       └── openai_model.py # OpenAI Server Model 适配器骨架
├── knowledge/
│   └── summarize.py        # 总结服务封装（依赖外部适配器）
├── obs/
│   └── events.py           # 事件结构与 append-only EventBus
└── persist/
    ├── export.py           # 图谱导出占位实现
    └── snapshot.py         # 快照/回滚管理
```

## 已实现的动作（Actions）

| Action 名称      | 功能概述 | 备注 |
|------------------|----------|------|
| `create_plan`    | 创建 `Plan` 节点并记录基础元数据 | 自动生成 `plan_<uuid>` | 
| `add_subtask`    | 在指定 Plan 下新增 `Subtask`，并建立 `CONTAINS` 边 | 需要 `plan_id` |
| `add_step`       | 为子任务新增 `Step` 节点，并与 Subtask 建立 `CONTAINS` 边 | 记录工具类型、输入等属性 |
| `add_data`       | 创建 `Data` 节点，记录数据来源信息 | 独立节点，无自动连边 |
| `link`           | 在任意两个节点之间显式创建边 | 支持指定 `EdgeType` 与属性 |
| `query`          | 基于 `QueryService` 查询图谱 | 支持 `by_type`、`timeline`、`neighbors` 三类 |

以下 action 已在路由器中占位，当前返回“未实现”提示：`run_step`、`summarize`、`record_note`、`auto_link`、`export_graph`、`snapshot`、`rollback`。

## 图谱模型

### 节点类型

- `Plan`：计划或实验项目，包含 `goal`、`owner`、`status` 等基础属性。
- `Subtask`：Plan 下的任务拆解，维护优先级和状态。
- `Step`：具体执行步骤，记录执行工具、输入、状态等。
- `Data`：数据资产引用，可存储外部存储位置、格式等信息。
- `Result`、`Note`、`Agent`：已在 schema 中预留，尚未由动作生成。

所有节点统一具备 `labels`、`created_at`、`updated_at` 等通用字段。

### 边类型

目前 `GraphStore` 支持以下边枚举：`CONTAINS`、`USES_DATA`、`PRODUCES`、`DERIVES`、`DEPENDS_ON`、`CAUSED_BY`、`FOLLOWS`、`ASSOCIATED_WITH`。实际动作只自动创建 `CONTAINS`；其他关系将随着 auto-link/执行模块完善逐步启用。

## 查询能力

- `by_type(type, **filters)`：返回指定类型且满足属性过滤的节点列表。
- `timeline(scope, include)`：按时间字段排序返回节点，可限定 plan 范围与节点类型。
- `neighbors(node_id, hop, edge_types)`：遍历指定跳数内的邻居节点。

## 可观测性与扩展点

- `EventBus`：每次 action 执行都会写入一条事件，含时间戳、级别与消息。
- `StepRunner` + 执行适配器：提供注册第三方执行器的接口，但默认未注册任何具体实现。
- `SummaryService`、`GraphExporter`、`SnapshotManager`：提供知识总结、导出与快照的基础封装，逻辑待接入核心流程。

## 开发计划

短期目标包括：

1. 补全 `run_step`、`summarize` 等核心动作，并串联执行/总结模块。
2. 实现自动连边规则（时序、依赖、因果）。
3. 打通持久化导出与快照回滚能力。
4. 增加端到端使用示例与测试用例。

上述结构为后续迭代预留了明确的模块边界，可按需替换或扩展各层能力。
