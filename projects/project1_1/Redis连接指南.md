# Redis 连接指南

本指南将帮助你安装、配置和连接 Redis 数据库。

## 📋 目录

- [Redis 简介](#redis-简介)
- [安装 Redis](#安装-redis)
- [启动 Redis 服务](#启动-redis-服务)
- [配置 Redis 连接](#配置-redis-连接)
- [测试连接](#测试连接)
- [在项目中使用 Redis](#在项目中使用-redis)
- [常见问题](#常见问题)

---

## 🔍 Redis 简介

Redis 是一个开源的内存数据结构存储系统，可以用作：
- **缓存**：存储工具调用结果，提高响应速度
- **会话管理**：存储用户会话数据
- **消息队列**：处理异步任务

**注意**：Redis 在本项目中是**可选的**，不安装 Redis 也能正常运行，只是缓存功能会被禁用。

---

## 📦 安装 Redis

### macOS

使用 Homebrew 安装：

```bash
# 安装 Redis
brew install redis

# 验证安装
redis-server --version
```

### Linux (Ubuntu/Debian)

```bash
# 更新包列表
sudo apt-get update

# 安装 Redis
sudo apt-get install redis-server

# 验证安装
redis-server --version
```

### Windows

1. 下载 Redis for Windows: https://github.com/microsoftarchive/redis/releases
2. 解压并运行 `redis-server.exe`

### Docker（推荐，跨平台）

```bash
# 运行 Redis 容器
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:latest

# 验证运行
docker ps | grep redis
```

---

## 🚀 启动 Redis 服务

### macOS (Homebrew)

```bash
# 启动 Redis（前台运行）
redis-server

# 或作为后台服务启动
brew services start redis

# 停止 Redis
brew services stop redis

# 查看 Redis 状态
brew services list | grep redis
```

### Linux (systemd)

```bash
# 启动 Redis
sudo systemctl start redis-server

# 设置开机自启
sudo systemctl enable redis-server

# 查看状态
sudo systemctl status redis-server

# 停止 Redis
sudo systemctl stop redis-server
```

### Docker

```bash
# 启动容器
docker start redis

# 停止容器
docker stop redis

# 查看日志
docker logs redis
```

---

## ⚙️ 配置 Redis 连接

### 1. 在 `.env` 文件中配置

编辑项目根目录的 `.env` 文件，添加以下配置：

```bash
# Redis 配置（可选）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### 2. 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `REDIS_HOST` | Redis 服务器地址 | `localhost` |
| `REDIS_PORT` | Redis 端口 | `6379` |
| `REDIS_DB` | 数据库编号（0-15） | `0` |
| `REDIS_PASSWORD` | Redis 密码（如果设置了） | 空（无密码） |

### 3. 如果 Redis 设置了密码

如果你的 Redis 服务器设置了密码，需要在 `.env` 中配置：

```bash
REDIS_PASSWORD=your_redis_password
```

---

## 🧪 测试连接

### 方法1：使用 Redis CLI

```bash
# 连接到 Redis
redis-cli

# 测试连接
ping
# 应该返回: PONG

# 设置一个测试值
set test_key "Hello Redis"

# 获取值
get test_key
# 应该返回: "Hello Redis"

# 退出
exit
```

### 方法2：使用 Python 测试

创建测试脚本 `test_redis.py`：

```python
import os
import redis
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

try:
    # 创建 Redis 客户端
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        password=os.getenv('REDIS_PASSWORD'),
        socket_timeout=5
    )

    # 测试连接
    redis_client.ping()
    print("✅ Redis 连接成功！")

    # 测试读写
    redis_client.set('test_key', 'test_value')
    value = redis_client.get('test_key')
    print(f"✅ 测试读写成功: {value.decode('utf-8')}")

    # 清理测试数据
    redis_client.delete('test_key')
    print("✅ 测试完成")

except redis.ConnectionError:
    print("❌ Redis 连接失败")
    print("💡 请确保 Redis 服务正在运行")
except Exception as e:
    print(f"❌ 错误: {str(e)}")
```

运行测试：

```bash
python test_redis.py
```

### 方法3：使用项目环境设置脚本

```bash
python scripts/setup_environment.py
```

脚本会自动检查 Redis 连接。

---

## 💻 在项目中使用 Redis

### 1. 基本使用示例

```python
import redis
from config.settings import settings

# 创建 Redis 客户端
redis_client = redis.Redis(
    host=settings.redis.redis_host,
    port=settings.redis.redis_port,
    db=settings.redis.redis_db,
    password=settings.redis.redis_password,
    decode_responses=True  # 自动解码为字符串
)

# 设置缓存
redis_client.set('key', 'value', ex=3600)  # 过期时间 3600 秒

# 获取缓存
value = redis_client.get('key')

# 删除缓存
redis_client.delete('key')
```

### 2. 缓存工具调用结果

```python
import json
import hashlib
from config.settings import settings

def get_cache_key(query: str, tool_name: str) -> str:
    """生成缓存键"""
    key_string = f"{tool_name}:{query}"
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_tool_result(query: str, tool_name: str, result: dict, ttl: int = 3600):
    """缓存工具调用结果"""
    try:
        redis_client = redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            db=settings.redis.redis_db,
            password=settings.redis.redis_password
        )
        cache_key = get_cache_key(query, tool_name)
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(result)
        )
    except Exception as e:
        print(f"缓存失败: {e}")

def get_cached_result(query: str, tool_name: str) -> dict:
    """获取缓存的工具结果"""
    try:
        redis_client = redis.Redis(
            host=settings.redis.redis_host,
            port=settings.redis.redis_port,
            db=settings.redis.redis_db,
            password=settings.redis.redis_password
        )
        cache_key = get_cache_key(query, tool_name)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        print(f"获取缓存失败: {e}")
    return None
```

---

## ❓ 常见问题

### Q1: Redis 连接失败

**错误信息**：
```
redis.ConnectionError: Error connecting to Redis
```

**解决方案**：
1. 确认 Redis 服务正在运行：
   ```bash
   # macOS
   brew services list | grep redis

   # Linux
   sudo systemctl status redis-server
   ```

2. 检查端口是否被占用：
   ```bash
   lsof -i :6379
   ```

3. 检查防火墙设置

4. 验证连接配置是否正确

### Q2: Redis 密码认证失败

**错误信息**：
```
redis.AuthenticationError: invalid password
```

**解决方案**：
1. 检查 `.env` 文件中的 `REDIS_PASSWORD` 是否正确
2. 如果 Redis 没有设置密码，确保 `REDIS_PASSWORD` 为空或删除该配置项

### Q3: 如何查看 Redis 中的数据

```bash
# 连接到 Redis CLI
redis-cli

# 列出所有键
keys *

# 查看特定键的值
get your_key

# 查看键的类型
type your_key

# 查看键的过期时间
ttl your_key

# 查看数据库大小
dbsize
```

### Q4: 如何清空 Redis 数据

```bash
# 连接到 Redis CLI
redis-cli

# 清空当前数据库
flushdb

# 清空所有数据库（谨慎使用！）
flushall
```

### Q5: Redis 占用内存过大

```bash
# 查看内存使用情况
redis-cli info memory

# 设置最大内存限制（在 redis.conf 中）
maxmemory 256mb
maxmemory-policy allkeys-lru
```

---

## 🔧 高级配置

### 1. 修改 Redis 配置文件

**macOS (Homebrew)**:
```bash
# 编辑配置文件
nano /usr/local/etc/redis.conf

# 或
nano /opt/homebrew/etc/redis.conf
```

**Linux**:
```bash
sudo nano /etc/redis/redis.conf
```

### 2. 常用配置项

```conf
# 绑定地址（0.0.0.0 表示允许所有IP连接）
bind 0.0.0.0

# 端口
port 6379

# 设置密码
requirepass your_password

# 最大内存
maxmemory 256mb

# 内存淘汰策略
maxmemory-policy allkeys-lru

# 持久化（RDB）
save 900 1
save 300 10
save 60 10000
```

### 3. 重启 Redis 使配置生效

```bash
# macOS
brew services restart redis

# Linux
sudo systemctl restart redis-server
```

---

## 📚 参考资源

- [Redis 官方文档](https://redis.io/docs/)
- [Redis Python 客户端文档](https://redis-py.readthedocs.io/)
- [Redis 命令参考](https://redis.io/commands/)

---

## ✅ 验证清单

完成以下步骤后，Redis 应该可以正常连接：

- [ ] Redis 已安装
- [ ] Redis 服务正在运行
- [ ] `.env` 文件中配置了 Redis 连接信息
- [ ] 使用 `redis-cli ping` 测试成功
- [ ] 使用 Python 脚本测试连接成功
- [ ] 运行 `python scripts/setup_environment.py` 显示 "✅ Redis连接成功"

---

**提示**：如果不需要缓存功能，可以不安装 Redis，项目仍然可以正常运行。
