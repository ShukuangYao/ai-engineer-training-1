# 测试文档

## 📋 测试概述

本项目包含完整的单元测试，覆盖核心功能模块。

## 🧪 测试文件

### 1. `test_qa_agent.py`
测试问答代理（QAAgent）的核心功能：
- ✅ 代理初始化
- ✅ 方法存在性检查
- ✅ 对话历史管理
- ✅ 清空对话历史
- ✅ 结束会话
- ⚠️ 对话处理（需要更复杂的 mock）
- ⚠️ 工具调用（需要更复杂的 mock）

### 2. `test_tools.py`
测试工具模块：
- ✅ 天气查询工具（AmapWeatherTool）
- ✅ 信息搜索工具（TavilySearchTool）
- ✅ 工具参数模式（tool_schemas）

### 3. `conftest.py`
Pytest 配置文件，提供：
- 测试环境设置
- 通用 fixtures
- Mock 配置

## 🚀 运行测试

### 运行所有测试

```bash
pytest tests/
```

### 运行特定测试文件

```bash
pytest tests/test_qa_agent.py
pytest tests/test_tools.py
```

### 运行特定测试类

```bash
pytest tests/test_qa_agent.py::TestQAAgent
```

### 运行特定测试方法

```bash
pytest tests/test_qa_agent.py::TestQAAgent::test_agent_initialization
```

### 详细输出

```bash
pytest tests/ -v
```

### 显示覆盖率

```bash
pytest tests/ --cov=agents --cov=tools --cov=config
```

## ⚙️ 测试配置

### 环境变量

测试会自动设置以下环境变量（在 `test_qa_agent.py` 中）：

```python
DEEPSEEK_API_KEY=test-deepseek-key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
AMAP_API_KEY=test-amap-key
TAVILY_API_KEY=test-tavily-key
```

### Mock 设置

测试使用 `unittest.mock` 来模拟：
- LLM 调用
- 工具执行
- API 请求
- 配置对象

## 📊 测试覆盖

### 已覆盖功能

- ✅ QAAgent 初始化
- ✅ 对话历史管理
- ✅ 工具初始化
- ✅ 错误处理
- ✅ 工具参数验证

### 需要改进的测试

- ⚠️ 完整的对话流程测试（需要更复杂的 mock）
- ⚠️ 工具调用的端到端测试
- ⚠️ 集成测试（需要真实 API 密钥）

## 🔧 编写新测试

### 测试模板

```python
import pytest
from unittest.mock import Mock, patch

class TestYourModule:
    """你的模块测试类"""

    @pytest.fixture
    def your_fixture(self):
        """测试用的 fixture"""
        return YourObject()

    def test_your_function(self, your_fixture):
        """测试你的函数"""
        result = your_fixture.your_function()
        assert result is not None
```

### 测试最佳实践

1. **使用 fixtures**: 复用测试对象
2. **Mock 外部依赖**: 避免真实 API 调用
3. **测试边界情况**: 空输入、错误处理等
4. **清晰的断言**: 使用有意义的错误消息
5. **测试隔离**: 每个测试应该独立运行

## ⚠️ 注意事项

1. **权限问题**: 测试会自动 mock `.env` 文件加载，避免权限错误
2. **API 密钥**: 测试使用模拟的 API 密钥，不会调用真实 API
3. **测试隔离**: 每个测试用例都是独立的，不会相互影响
4. **警告信息**: Pydantic 的弃用警告是正常的，不影响测试

## 📝 测试示例

### 基本测试

```python
def test_agent_initialization(agent):
    """测试代理初始化"""
    assert agent is not None
    assert agent.session_id == "test-session-123"
```

### Mock 测试

```python
@patch('tools.amap_weather_tool.requests.get')
def test_get_weather_success(mock_get, weather_tool):
    """测试成功获取天气"""
    mock_get.return_value.json.return_value = {...}
    result = weather_tool.get_weather("北京")
    assert result['success'] is True
```

## 🔍 调试测试

### 运行单个测试并显示输出

```bash
pytest tests/test_qa_agent.py::TestQAAgent::test_agent_initialization -v -s
```

### 在失败时进入调试器

```bash
pytest tests/ --pdb
```

### 显示详细错误信息

```bash
pytest tests/ -v --tb=long
```

## 📚 参考资料

- [Pytest 文档](https://docs.pytest.org/)
- [unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)
- [测试最佳实践](https://docs.python.org/3/library/unittest.html)
