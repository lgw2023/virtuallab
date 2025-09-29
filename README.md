# virtuallab
我想基于networkX的图谱结构构件一个VirtualLab，用来模拟数据分析实验和实验结果知识整理的目的。

作为虚拟实验室的定位来说，对外暴露的方法和接口要简洁优雅，使得能够仅靠VirtualLab_tool(xxx)的多次调用和传参就能够完成：
- 计划创建
- 数据添加、
- 子任务拆解编排
- 子任务各实验步骤添加
- 实验步骤执行
- 结果总结
- 知识记录
- 自动更新检索全局图谱内容实现关联实验步骤或者结论的时序/逻辑顺序/因果关系
- 以及其他模拟虚拟实验室和知识管理的功能。 

我额外提供了：（1）一个Engineer智能体，该智能体接受文字信息输入，可自动调用各种工具完成分析任务并最终返回文字化的分析结果；（2）一个大模型端口，OpenAIServerModel，可进行聊天对话、function calling等通用能力。


下面先给出一个清晰可扩展的「VirtualLab」框架蓝图。先统一概念与模块边界，然后逐步把各模块落成代码。

# 1) 总体架构（分层）

* **Interface 层（对外唯一门面）**

  * `VirtualLab_tool(payload: dict) -> dict`
  * 纯函数式风格，所有操作通过 `action` + `params` 调用，返回标准化 `result` + `events`。
* **Orchestration 层（编排）**

  * 命令路由器（Action Router）
  * 事务/事件总线（Event Bus）
  * 权限与配额（可选）
* **Core Graph 层（基于 NetworkX）**

  * 全局图：`nx.MultiDiGraph`
  * 节点/边类型与属性标准（Plan / Subtask / Step / Data / Result / Note / Agent）
  * 版本与时序（时间戳、因果/依赖）
  * 自动连边规则（时序、逻辑、因果）
* **Execution 层（实验执行）**

  * Step 执行器（本地函数/外部工具/Agent）
  * `Engineer` 智能体适配器
  * `OpenAIServerModel` 适配器（聊天、函数调用）
  * 运行时上下文（输入/输出、重试、缓存、可重现性）
* **Knowledge 层（知识与总结）**

  * 结果总结器（LLM 或规则）
  * 知识记录本（Note 节点/边）
  * 图谱检索与查询（结构化 Query + 语义检索可选）
  * 自动更新与联想（根据新增节点/边触发）
* **Persistence 层（存储与导入导出）**

  * 内存态（初期）
  * 文件持久化（GraphML/JSON）
  * 快照/回滚
* **Observability 层（可观测性）**

  * 事件流（append-only）
  * 审计追踪（who/when/what）
  * 最小可视化（后续可加）

# 2) 图谱设计（Schema v0）

## 节点（`node[type=...]`）

* `Plan`: 计划（name, goal, created_at, owner, status）
* `Subtask`: 子任务（plan_id, name, status, priority, created_at）
* `Step`: 步骤（subtask_id, name, tool, inputs, status, created_at, executed_at, run_id）
* `Data`: 数据（payload_ref 或内联、format、source、hash、created_at）
* `Result`: 结果（summary, detail_ref, metrics, created_at, produced_by_step）
* `Note`: 知识/笔记（content, tags, created_at, linked_to）
* `Agent`: 执行体（kind ∈ {Engineer, OpenAIServerModel, LocalFunc}, config）

> 约定：所有节点有 `id`、`created_at`、`updated_at`、`labels`（list[str]）

## 边（`edge[type=...]`，有向多重边）

* `CONTAINS`：Plan→Subtask，Subtask→Step
* `USES_DATA`：Step→Data
* `PRODUCES`：Step→Result
* `DERIVES`：Result→Note（或 Note→Result，双向可选）
* `DEPENDS_ON`：Step→Step（逻辑依赖/前置）
* `CAUSED_BY`：Conclusion/Result/Note→Step（因果）
* `FOLLOWS`：Step/Result/Note 时序链接（自动）
* `ASSOCIATED_WITH`：任意语义关联（标签/实体/主题）

# 3) 外部门面：`VirtualLab_tool` 协议（v0）

所有调用均为：

```json
{
  "action": "<string>",
  "params": { ... },
  "options": { "return": ["graph_delta", "nodes", "edges", "events"] }
}
```

**核心 actions（首批）**

* `create_plan`: 创建计划
* `add_data`: 添加数据节点（支持内联/引用）
* `add_subtask`: 往计划里加子任务
* `add_step`: 给子任务加步骤（指定 tool=Engineer/OpenAIServerModel/Local）
* `run_step`: 执行某步骤（分派至相应适配器）
* `summarize`: 对 Result/Plan/Subtask 生成总结（Note 节点）
* `record_note`: 记录知识（Note）
* `link`: 显式建立边（任意两节点）
* `auto_link`: 基于规则/元数据自动连边（时序/逻辑/因果）
* `query`: 结构化检索（按 type/属性/拓扑）
* `export_graph`: 导出（graphml/json）
* `snapshot` / `rollback`: 快照管理

**返回结构（统一）**

```json
{
  "ok": true,
  "result": { ... },
  "events": [ { "ts": "...", "level": "info", "msg": "..." } ],
  "graph_delta": { "added_nodes": [...], "added_edges": [...], "updated": [...] }
}
```

# 4) 执行适配器

* **EngineerAdapter**

  * `run(text_input, tools=[]) -> text_output`
  * 将 Step.inputs（自然语言+上下文）喂给 Engineer 智能体，拿回文字化分析结果，产出 Result 节点。
* **OpenAIServerModelAdapter**

  * `chat(messages, functions=None) -> output`
  * 支持 function calling；用于总结/生成/推理。
* **LocalFuncAdapter**

  * 注册本地 Python 函数（如简单统计、清洗），便于无外部依赖的可重现步骤。

# 5) 自动连边（Auto-Link）策略 v0

* **时序**：同一 Subtask/Plan 内，按 `executed_at` 自动添加 `FOLLOWS`（Step/Result/Note）。
* **逻辑**：若 Step.B.inputs 引用了某 Step.A 的输出/Result.id，则添加 `DEPENDS_ON: A→B`。
* **因果**：若 Note/Result 的 `causal_ref` 指向某 Step，则 `CAUSED_BY`。
* **语义**（可选）：

  * 标签/关键词重叠度超过阈值 → `ASSOCIATED_WITH`
  * 由 LLM 做轻量归因提示（后续再加）。

# 6) Query 形态（结构化为主，便于 deterministic）

* `query.by_type(type, filters)`
* `query.paths(source_id, target_id, edge_types=[...])`
* `query.neighbors(id, hop=1, edge_types=...)`
* `query.timeline(scope={plan_id|subtask_id}, include=['Step','Result','Note'])`
* `query.causality(step_id)` 返回其直接/间接因果链

# 7) 事件与可观测性

* 统一 `Event`：`{ts, actor, action, target_ids, msg, extras}`
* 每次 `VirtualLab_tool` 调用都追加事件；便于审计/回放。

# 8) 最小交互示例（逻辑流程）

```python
VirtualLab_tool({"action":"create_plan", "params":{
  "name":"销售数据异常分析", "goal":"定位异常原因并形成复盘"
}})

VirtualLab_tool({"action":"add_data", "params":{
  "plan_id": "...", "name":"Q3 销售表", "payload_ref":"s3://...", "format":"csv"
}})

VirtualLab_tool({"action":"add_subtask", "params":{
  "plan_id":"...", "name":"异常检测"
}})

VirtualLab_tool({"action":"add_step", "params":{
  "subtask_id":"...", "name":"初步概览",
  "tool":"Engineer", "inputs":{
    "text":"请对Q3销售表做基础统计与异常点预判。数据位置见节点:<data_id>",
    "tools":["pandas","plot"]
  }
}})

VirtualLab_tool({"action":"run_step", "params":{
  "step_id":"..."
}})

VirtualLab_tool({"action":"summarize", "params":{
  "target_id":"<result_id>", "adapter":"OpenAIServerModel",
  "prompt":"将上述结果转为结构化要点，并输出可能的因果假设。"
}})

VirtualLab_tool({"action":"auto_link", "params":{
  "scope":{"plan_id":"..."},
  "rules":["temporal","logic","causal"]
}})

VirtualLab_tool({"action":"query", "params":{
  "kind":"timeline", "scope":{"plan_id":"..."},
  "include":["Step","Result","Note"]
}})
```

# 9) 落地计划（实现顺序）

1. **Core Graph & Schema + 门面函数骨架**（含内存存储、ID、时间戳、事件流）。
2. **基本 Actions**：`create_plan / add_subtask / add_step / add_data / link / query`。
3. **执行器框架**：`run_step` + `EngineerAdapter` & `OpenAIServerModelAdapter` 的接口定义（先打桩，后接真实现）。
4. **Result / Note / summarize & record_note**。
5. **auto_link 规则引擎 v0**（时序/逻辑/因果的最小规则）。
6. **导出/快照** 与 **可视化占位**。
7. **单元测试与最小示例**。

# 10) 代码组织建议

```
virtuallab/
  __init__.py
  api.py                 # VirtualLab_tool 门面
  router.py              # Action Router
  graph/
    model.py             # 节点/边 schema & helpers
    store.py             # Graph 管理（nx.MultiDiGraph）
    ids.py               # ID/时间戳工具
    rules.py             # auto_link 规则
    query.py             # 查询
  exec/
    runner.py            # StepRunner（统一执行流）
    adapters/
      engineer.py        # EngineerAdapter
      openai_model.py    # OpenAIServerModelAdapter
      local.py           # LocalFuncAdapter
  knowledge/
    summarize.py         # 总结器
  persist/
    export.py            # graphml/json
    snapshot.py
  obs/
    events.py            # 事件总线
```

