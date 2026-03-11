# `get_state_history(config)` 快照输出详解

对应代码：`p44-snapshot.ipynb` 中「获取状态历史（快照）」部分。

```python
state_history = list(graph.get_state_history(config))
for state in state_history:
    print(state)
```

---

## 一、这段代码在干什么？

- **`graph.get_state_history(config)`**：按当前 `config`（通常带 `thread_id`）从 **Checkpointer**（如 `MemorySaver`）里取出**该线程的所有检查点**，按时间从新到旧迭代。
- 每一个元素是一个 **`StateSnapshot`**，即「某一时刻图的状态 + 元信息」。
- 因为编译时传了 `checkpointer=memory`，所以每次节点执行完都会落盘一个检查点，历史就是这些检查点组成的快照序列。

---

## 二、快照的顺序：从新到旧

迭代顺序是**先最新、再往前**：

| 顺序 | 含义 | 对应本 notebook 的图 |
|------|------|----------------------|
| 第 1 个 | 最后一次检查点（图跑完后的状态） | node2 执行完 → END 之前 |
| 第 2 个 | 倒数第二次 | node1 执行完，即将进 node2 |
| 第 3 个 | 再往前 | __start__ 执行完，即将进 node1 |
| 第 4 个 | 最早 | 刚注入输入，尚未执行任何节点 |

所以「第一个快照 = 最终状态」，「最后一个快照 = 初始输入后的状态」。

---

## 三、StateSnapshot 里每个字段是什么？

每个 `state` 都是类似这样的对象（下面用你 notebook 里的实际输出对应说明）：

### 1. `values` —— 当前状态的内容

- **含义**：这一拍快照下，图的**完整状态**（就是你定义的 `AgentState` 里那些键）。
- **例**：
  - 第 1 个快照：`values={'messages': ['Initial input', 'Hello from node1 at step 1', 'Goodbye from node2 at step 2'], 'step_count': 2}` → 图结束时的状态。
  - 第 3 个快照：`values={'messages': ['Initial input'], 'step_count': 0}` → 刚进入图、还没跑 node1 时的状态。

用 `snapshot.values` 可以拿到「当时」的 `messages`、`step_count` 等，用于回放或调试。

---

### 2. `next` —— 从这个快照出发，下一步会执行谁

- **含义**：从当前检查点**接着跑**时，将要执行的**节点名**（或边目标）。
- **类型**：一般是元组，如 `()`、`('node2',)`、`('node1',)`；空元组 `()` 表示没有下一步（例如已到 END）。
- **例**：
  - 第 1 个快照：`next=()` → 已经结束，没有下一个节点。
  - 第 2 个快照：`next=('node2',)` → 下一步会执行 node2。
  - 第 4 个快照：`next=('__start__',)` → 下一步是图的入口 __start__（然后会进 node1）。

用来知道「如果从这个快照恢复，图会先跑哪个节点」。

---

### 3. `config` —— 这个快照对应的运行时配置

- **含义**：生成这个检查点时用的 **config**，里面主要是可配置项。
- **典型内容**：
  - `thread_id`：会话/线程 ID，和你在 `config = {"configurable": {"thread_id": "123"}}` 里的一致。
  - `checkpoint_ns`：检查点命名空间，一般空串。
  - `checkpoint_id`：**这个快照的唯一 ID**（UUID）。用 `config["configurable"]["checkpoint_id"]` 指定它，就可以从**这个快照**恢复或继续跑（见 notebook 里「根据 checkpoint_id 重新启动」的 cell）。

不同快照的 `checkpoint_id` 不同，所以可以精确「回到某一拍状态」。

---

### 4. `metadata` —— 来源与步数

- **含义**：描述这个快照是**怎么来的**、是第几步。
- **常见字段**：
  - **`source`**：
    - `'input'`：由**输入**产生的初始检查点（还没进任何节点）。
    - `'loop'`：图**执行过程中**某节点执行完后的检查点。
  - **`step`**：整数步数。`-1` 表示「输入刚写入」；`0, 1, 2, ...` 表示第 0、1、2… 次「图循环」后的状态（通常每执行完一个节点就一步）。
  - **`parents`**：父检查点等依赖信息，简单图里常为空。

结合 `values` 和 `metadata.step` 可以画出「第几步时状态长什么样」。

---

### 5. `created_at` —— 创建时间

- **含义**：这个检查点**写入**的时间（ISO 格式）。
- 用于排查顺序、延迟或并发问题。

---

### 6. `parent_config` —— 上一个检查点的 config

- **含义**：**产生当前快照的那一步**所对应的「上一步」检查点的 config（包含那个检查点的 `checkpoint_id`）。
- **用途**：顺着 `parent_config` 可以往前追溯整条检查点链；第一个快照（输入）的 `parent_config` 为 `None`。

---

### 7. `tasks` —— 从当前快照「已经/正在」执行的任务

- **含义**：从**当前**检查点出发，**已经调度或执行完**的 **PregelTask**。每个任务代表「跑哪个节点、路径、结果」。
- **典型内容**（一个 `PregelTask` 里常见字段）：
  - **`name`**：节点名，如 `'node1'`、`'node2'`、`'__start__'`。
  - **`path`**：在图中的路径，如 `('__pregel_pull', 'node1')`。
  - **`result`**：该节点**返回的更新**（会合并进下一拍的 `values`）。例如 `result={'messages': ['Hello from node1 at step 1'], 'step_count': 1}`。

可以这样理解：**当前快照 = 上一拍状态 + 本拍执行的 `tasks` 的 result 合并后的结果**。所以：
- 第 4 个快照（输入后）：`tasks` 里是 `__start__` 的 task，其 `result` 就是把初始输入写进状态。
- 第 3 个快照：`tasks` 里是 node1 的 task，`result` 是 node1 返回的 `messages` 和 `step_count`。
- 第 2 个快照：`tasks` 里是 node2 的 task，依此类推。

---

### 8. `interrupts` —— 中断信息

- **含义**：若在图里配置了 **interrupt**（例如 `interrupt_before=["node2"]`），在中断点会在这里记录。
- 你当前这个简单图没有中断，所以一般是空元组 `()`。

---

## 四、和你 notebook 里 4 个快照的对应关系（简要）

| 打印顺序 | step | source | values 概览 | next | 含义 |
|----------|------|--------|-------------|------|------|
| 1 | 2 | loop | messages 有 3 条，step_count=2 | () | node2 刚跑完，图结束 |
| 2 | 1 | loop | messages 2 条，step_count=1 | ('node2',) | node1 跑完，马上要跑 node2 |
| 3 | 0 | loop | messages 1 条，step_count=0 | ('node1',) | __start__ 已写入输入，马上跑 node1 |
| 4 | -1 | input | messages=[]（或初始） | ('__start__',) | 仅输入写入，尚未执行任何节点 |

---

## 五、常用用法小结

1. **看状态演变**：遍历 `state_history`，看每个 `snapshot.values`（和 `metadata.step`），就能复现「每一步之后状态长什么样」。
2. **回放到某一拍**：取某个 `snapshot.config`（或其中的 `checkpoint_id`），作为新的 `config` 调用 `graph.invoke(..., config=config)`，就会从那个快照继续跑（见 notebook 里带 `checkpoint_id` 的 cell）。
3. **看「下一步是谁」**：用 `snapshot.next` 知道从该快照恢复时会先执行哪个节点。
4. **看某一步是谁改的**：看该快照的 `tasks`，里面任务的 `name` 和 `result` 就是「谁执行了、改了什么」。

这样，**快照 = 状态在某时刻的完整拷贝 + 从哪来、到哪去、谁造成的**，用于调试、回放和从中间状态恢复执行。
