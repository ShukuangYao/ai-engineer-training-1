# AgentExecutor 与 create_react_agent 讲解

## 一、整体关系

```
create_react_agent(llm, tools, prompt)  →  得到一个 agent（只负责「想一步、出解析」）
                    ↓
AgentExecutor(agent=agent, tools=tools) →  执行循环：调用 agent → 跑 tool → 把结果塞回 agent → 直到出 Final Answer
                    ↓
agent_executor.invoke({"input": "问题"})  →  返回最终答案（和可选的 intermediate_steps）
```

- **create_react_agent**：造一个 **ReACT 风格的 agent**，输入是当前状态，输出是「下一步要做什么」（Thought + Action + Action Input，或 Final Answer）。
- **AgentExecutor**：拿着这个 agent 和 tools，**反复执行**「问 agent → 若需要就调 tool → 把 Observation 写回 prompt → 再问 agent」，直到 agent 给出 Final Answer 或达到 `max_iterations`。

---

## 二、create_react_agent

**作用**：根据你给的 **LLM、tools、prompt 模板**，拼出一个 **ReACT agent**（一个 LangChain Runnable）。这个 agent **只做一件事**：根据当前输入和 `agent_scratchpad`，输出**一段文本**，里面要么是「Thought / Action / Action Input」，要么是「Thought / Final Answer」。

**典型用法**：

```python
agent = create_react_agent(llm, tools, prompt)
```

**参数**：

| 参数 | 含义 |
|------|------|
| **llm** | 用来「思考」的模型（OpenAI、Tongyi、ChatOpenAI 等），按 prompt 生成 Thought/Action 或 Final Answer。 |
| **tools** | 可选工具列表（`Tool` 或 @tool），会出现在 prompt 的 `{tools}`、`{tool_names}` 里，供模型选择。 |
| **prompt** | `PromptTemplate`，必须包含占位符：`{tools}`、`{tool_names}`、`{input}`、`{agent_scratchpad}`。 |

**Prompt 里各占位符**：

- **`{tools}`**：工具名 + 描述，模型用来看「能调用什么」。
- **`{tool_names}`**：工具名列表（如 `TrackPackage, CalculateShipping, CheckInventory`），用来约束 Action 填哪个。
- **`{input}`**：用户问题。
- **`{agent_scratchpad}`**：**历史推理过程**：之前几轮的 `Thought` / `Action` / `Action Input` / `Observation` 拼在一起，让模型知道「已经做过什么、观察到什么」，从而决定下一步是再调工具还是给 Final Answer。

也就是说，**create_react_agent 只定义「单步推理」**：给当前状态 → 模型输出一段 ReACT 格式文本。**谁去解析这段文本、真的调 tool、再把 Observation 写回 scratchpad，是 AgentExecutor 的事**。

---

## 三、AgentExecutor

**作用**：**执行 agent 的循环**：  
1）用当前 `input` + `agent_scratchpad` 调用 agent；  
2）解析 agent 输出（是 Action 还是 Final Answer）；  
3）若是 Action → 调对应 tool，得到 Observation；  
4）把这一轮的 Thought / Action / Action Input / Observation 追加到 `agent_scratchpad`；  
5）重复 1～4，直到解析到 Final Answer 或达到 `max_iterations`。

**常用参数**：

| 参数 | 含义 |
|------|------|
| **agent** | 由 create_react_agent 得到的 agent，负责「想一步」。 |
| **tools** | 与 agent 里用到的保持一致，Executor 根据 agent 输出的 Action 名去查并执行。 |
| **verbose** | 是否在控制台打印每一轮的 Thought / Action / Observation。 |
| **handle_parsing_errors** | agent 输出没按 ReACT 格式（或解析失败）时，是否把错误信息塞回模型让它重试，而不是直接抛异常。 |
| **max_iterations** | 最多执行多少轮（一轮 = 一次 Thought→Action→Observation），防止死循环。 |
| **return_intermediate_steps** | invoke 的返回里是否带上每一轮的 (agent_action, observation)，便于展示或调试。 |

**一次 invoke 的流程可以简化为**：

```
用户 input
  → agent(prompt 里 scratchpad 为空) → 输出 "Thought: ... Action: TrackPackage Action Input: SF123456"
  → Executor 解析出 tool=TrackPackage, input=SF123456
  → 执行 tools 里 TrackPackage("SF123456") → Observation: "快递单号: SF123456, 状态: 运输中..."
  → 把 "Thought: ... Action: ... Observation: ..." 追加到 agent_scratchpad
  → 再次 agent(同一 input，scratchpad 有内容) → 输出 "Thought: 我已获取到... Final Answer: 快递单号 SF123456 当前状态为..."
  → Executor 解析出 Final Answer，结束，返回 {"output": "快递单号 SF123456..."}
```

---

## 四、与 p35-ReACT.ipynb 的对应关系

- **create_react_agent(model, tools, prompt)**：用你的 `template`（含 `{tools}` / `{tool_names}` / `{input}` / `{agent_scratchpad}`）和物流三个 Tool，造出「单步 ReACT agent」。
- **AgentExecutor(agent=..., tools=tools, verbose=True, ...)**：负责循环调用这个 agent、解析输出、执行 TrackPackage / CalculateShipping / CheckInventory、把 Observation 写回 scratchpad，直到出现 Final Answer，并把最终答案（和可选的 intermediate_steps）返回给你。

**一句话**：**create_react_agent = 造出「按 ReACT 格式想一步」的 agent；AgentExecutor = 用这个 agent + tools 做循环执行，直到得到最终答案。**
