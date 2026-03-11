# p44-snapshot 测试运行流程图（单元格 83–97）

对应代码：

```python
config = {"configurable": {"thread_id": "123"}}
app.invoke({"messages": ["abc123"]}, config=config)
history = app.get_state_history(config)
for snapshot in history:
    print("Messages:", snapshot.values["messages"])
    print("Retry count:", snapshot.values.get("retry_count", 0))
    print("---")
```

---

## 一、单元格整体执行流程

```mermaid
flowchart LR
    A["设置 config<br/>thread_id=123"] --> B["app.invoke(<br/>messages=['abc123'],<br/>config)"]
    B --> C["get_state_history(config)"]
    C --> D["for snapshot in history"]
    D --> E["打印每条快照的<br/>messages 与 retry_count"]
```

---

## 二、`app.invoke` 内部：图执行路径（输入 `"abc123"` 非法）

输入 `"abc123"` 不是 10～12 位数字，走 **invalid → handle_invalid → receive_input → query_order → END**。

```mermaid
flowchart TB
    Start([START]) --> Validate["validate_input<br/>校验最后一条消息"]
    Validate -->|"合法<br/>10~12位数字"| Query["query_order<br/>查订单并返回结果"]
    Validate -->|"非法<br/>abc123"| Invalid["handle_invalid<br/>retry_count=1<br/>追加提示消息"]
    Invalid --> Check{"retry_count < 2?"}
    Check -->|"是"| Receive["receive_input<br/>读最后一条消息"]
    Check -->|"否"| End1([END])
    Receive --> Query
    Query --> End2([END])

    style Invalid fill:#f9f,stroke:#333
    style Receive fill:#bbf,stroke:#333
    style Query fill:#bfb,stroke:#333
```

本次运行实际路径（粗线）：**START → validate_input → invalid → handle_invalid → receive_input → query_order → END**。

---

## 三、状态与快照对应关系（输入 `"abc123"`）

`get_state_history` 按**从新到旧**顺序迭代，每次 `invoke` 会留下多个检查点快照：

```mermaid
flowchart TB
    subgraph 快照顺序["get_state_history 迭代顺序（先新后旧）"]
        S1["① 最新快照<br/>messages: abc123 + 订单号不合法 + 订单状态已发货<br/>retry_count: 1"]
        S2["② 上一拍<br/>messages: abc123 + 订单号不合法<br/>retry_count: 1"]
        S3["③ 更早<br/>messages: abc123<br/>retry_count: 0"]
        S4["④ 最初<br/>messages: []<br/>retry_count: 0"]
    end
    S1 --> S2 --> S3 --> S4
```

| 快照 | 对应时机 | messages | retry_count |
|------|----------|----------|-------------|
| ① | query_order 执行完 | [abc123, 订单号不合法..., 订单状态: 已发货] | 1 |
| ② | handle_invalid 执行完 | [abc123, 订单号不合法...] | 1 |
| ③ | 输入刚写入 / validate 前 | [abc123] | 0 |
| ④ | 图入口（尚未处理输入） | [] | 0 |

---

## 四、合在一起：单元格 + 图 + 快照

```mermaid
flowchart TB
    subgraph 单元格["单元格执行"]
        C1["config = thread_id 123"]
        C2["app.invoke(messages=['abc123'])"]
        C3["history = get_state_history(config)"]
        C4["for snapshot: print messages, retry_count"]
    end
    subgraph 图执行["invoke 时图内部"]
        G1([START]) --> G2["validate_input"]
        G2 -->|invalid| G3["handle_invalid"]
        G3 --> G4["receive_input"]
        G4 --> G5["query_order"]
        G5 --> G6([END])
    end
    subgraph 快照["检查点快照 ①→④"]
        H1["① 最终状态"]
        H2["② handle_invalid 后"]
        H3["③ 输入后"]
        H4["④ 初始"]
    end

    C1 --> C2
    C2 --> C3
    C3 --> C4
    C2 -.->|内部执行| G1
    G6 -.->|写入检查点| H1
    C3 -.->|读取| H1
```

说明：先执行 `invoke`，图按上面路径跑并写入检查点；再执行 `get_state_history` 按 thread_id 读出这些快照，循环打印每条快照的 `messages` 和 `retry_count`。
