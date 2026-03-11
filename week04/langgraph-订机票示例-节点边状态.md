# 用「订机票系统」理解 LangGraph：状态、节点、边

## 一、把流程想成一张图

订机票大致是：**查航班 → 选航班 → 填乘机人 → 付款 → 出票**，中间可能**退回上一步**或**分支**（例如付款失败重试、取消）。  
LangGraph 就是把这种流程画成一张**有状态的图**：**状态**是手里拿着的「所有信息」，**节点**是「做一件事」，**边**是「做完之后下一步去哪」。

---

## 二、状态（State）—— 手里有什么信息

**状态 = 整张图共享的一份「当前进度 + 数据」**，每个步骤都从状态里读、再往状态里写。

订机票流程里，可以定义状态里要带哪些东西：

| 字段 | 含义 | 谁写入 |
|------|------|--------|
| `query` | 用户查询（出发地、目的地、日期） | 入口输入 |
| `flights` | 搜到的航班列表 | 查航班 |
| `selected_flight` | 用户选中的航班 | 选航班 |
| `passenger` | 乘机人姓名、证件号 | 填乘机人 |
| `payment_ok` | 是否付款成功 | 付款 |
| `order_id` | 订单号（出票后才有） | 出票 |
| `current_step` | 当前到哪一步（方便路由） | 各节点 |

用 TypedDict 定义就是「状态长什么样」：

```python
from typing import TypedDict

class BookingState(TypedDict):
    query: dict           # 出发地、目的地、日期
    flights: list         # 航班列表
    selected_flight: dict  # 选中的航班
    passenger: dict       # 乘机人信息
    payment_ok: bool      # 是否付款成功
    order_id: str         # 订单号
    current_step: str      # 当前步骤名，用于路由
```

- **读**：节点里用 `state["query"]`、`state["flights"]` 等。
- **写**：节点 `return {"flights": [...]}`，LangGraph 会把这些键**合并**进当前状态，再传给下一个节点。

所以：**定义状态 = 约定「图在跑的时候，共享数据里有哪些字段、怎么更新」。**

---

## 三、节点（Nodes）—— 每一步做什么

**节点 = 一个函数**：输入是当前**状态**，输出是**要写回状态的一小部分**（以及可选的边名）。

订机票可以拆成 5 个节点：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  查航班     │ →  │  选航班     │ →  │ 填乘机人    │ →  │  付款       │ →  │  出票       │
│ search      │    │ select      │    │ passenger   │    │ pay         │    │ issue_ticket │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

每个节点只做一件事，例如：

- **search_flights(state)**  
  - 读：`state["query"]`  
  - 写：`return {"flights": [...], "current_step": "searched"}`

- **select_flight(state)**  
  - 读：`state["flights"]` + 用户选择（可从 state 或外部输入）  
  - 写：`return {"selected_flight": {...}, "current_step": "selected"}`

- **fill_passenger(state)**  
  - 读：无/或上一步结果  
  - 写：`return {"passenger": {...}, "current_step": "passenger_ok"}`

- **pay(state)**  
  - 读：`selected_flight`、`passenger`  
  - 写：`return {"payment_ok": True/False, "current_step": "paid"}`

- **issue_ticket(state)**  
  - 读：`selected_flight`、`passenger`、`payment_ok`  
  - 写：`return {"order_id": "ORD123", "current_step": "done"}`

节点**不关心**「上一步是谁、下一步是谁」，只关心：**当前状态里有什么 → 算完 → 要更新状态的哪几项**。  
「下一步去哪」由**边**决定。

---

## 四、边（Edges）—— 做完这一步，下一步去哪

**边 = 从当前节点到下一个节点的「怎么走」**。分两种：

### 1. 固定边（顺序下一步）

- 查航班 → **一定**去「选航班」  
- 填乘机人 → **一定**去「付款」  
- 出票 → **一定**结束  

在 LangGraph 里就是：`add_edge("search_flights", "select_flight")` 等。

### 2. 条件边（根据状态决定下一步）

典型：**付款**之后要根据 `payment_ok` 决定：

- `payment_ok == True` → 去「出票」
- `payment_ok == False` → 回「付款」重试，或去「填乘机人」改信息

这时会写一个**路由函数**，读状态，返回**下一个节点的名字**：

```python
def after_pay(state):
    if state.get("payment_ok"):
        return "issue_ticket"   # 付款成功 → 出票
    else:
        return "pay"            # 付款失败 → 再付一次（或 "fill_passenger"）
```

图上就加一条**条件边**：从 `pay` 出发，用 `after_pay(state)` 的结果决定去哪个节点。

---

## 五、合在一起：状态 + 节点 + 边

一句话对应关系：

| 概念 | 订机票里的含义 |
|------|----------------|
| **状态** | 当前查询、航班列表、选中航班、乘机人、是否付成功、订单号…… 整张图共享的「进度+数据」 |
| **节点** | 查航班、选航班、填乘机人、付款、出票 —— 每一步的**一个函数**，只读状态、只写要更新的字段 |
| **边** | 查完→选、选完→填、填完→付、付成功→出票、付失败→重付 等，**固定边**管顺序，**条件边**管分支 |

流程可以抽象成：

```
                    ┌──────────────┐
  inputs ──────────►│ query       │
  (query)           │ flights=[]  │
                    │ ...        │
                    └──────┬──────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ search_      │  节点：查航班
                    │ flights      │  写：flights, current_step
                    └──────┬──────┘
                           │ 固定边
                           ▼
                    ┌──────────────┐
                    │ select_      │  节点：选航班
                    │ flight       │  写：selected_flight
                    └──────┬──────┘
                           │ 固定边
                           ▼
                    ┌──────────────┐
                    │ fill_        │  节点：填乘机人
                    │ passenger    │  写：passenger
                    └──────┬──────┘
                           │ 固定边
                           ▼
                    ┌──────────────┐
                    │ pay          │  节点：付款
                    └──────┬──────┘  写：payment_ok
                           │ 条件边：after_pay(state)
                    ┌──────┴──────┐
                    ▼             ▼
            payment_ok?      payment_ok?
               True              False
                    │             │
                    ▼             ▼
            ┌──────────────┐  ┌──────────────┐
            │ issue_ticket │  │ pay (重试)   │
            │ 写：order_id │  │ 或 fill_     │
            └──────┬──────┘  │ passenger    │
                   │         └──────────────┘
                   ▼
                  END
```

---

## 六、和 RAG 图类比（你 notebook 里的图）

| 订机票 | p42-langgraph-0RAG |
|--------|---------------------|
| 状态里的 query、flights、selected_flight… | 状态里的 question、documents、generation、web_search… |
| 节点 search / select / pay / issue_ticket | 节点 retrieve / generate / web_search / grade_generation… |
| 条件边：付款成功→出票、失败→重付 | 条件边：路由→websearch 或 vectorstore；评分→继续生成或 websearch |

本质一样：**状态**是图里传递的「上下文」，**节点**是「读状态、干一件事、写回状态」，**边**是「下一步走到哪个节点」（固定或按条件）。

---

## 七、小结

- **状态**：整张图共享的「当前有哪些数据、进度到哪了」；先定义 TypedDict，节点只读写其中一部分。
- **节点**：一个函数 `(state) -> 部分状态更新`，只负责一步逻辑，不决定下一步去哪。
- **边**：固定边 = 顺序下一步；条件边 = 根据 `state` 决定下一个节点名。

订机票系统里：**状态 = 订单进度 + 航班/乘机人/付款结果**，**节点 = 查、选、填、付、出票**，**边 = 顺序 + 付款成功/失败的分支**。用这个例子套到任何「多步、有分支」的流程（RAG、客服、审批）都是一样的思路。
