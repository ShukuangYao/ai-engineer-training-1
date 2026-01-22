import sqlite3
import json
import uuid
import datetime
from typing import Dict, Any, List

class ExecutionLogger:
    """基于 SQLite 的执行日志记录器。"""
    
    def __init__(self, db_path: str = "execution_logs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 创建 traces 表 (存储一次完整执行的元数据)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    dsl_file TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    input TEXT,
                    output TEXT
                )
            """)
            # 创建 events 表 (存储每个步骤的详细日志)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    timestamp TIMESTAMP,
                    step_name TEXT,
                    event_type TEXT,
                    details TEXT,
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
                )
            """)
            conn.commit()

    def start_trace(self, dsl_file: str, input_data: Dict[str, Any]) -> str:
        """开始一次新的追踪，返回 trace_id。"""
        trace_id = str(uuid.uuid4())
        start_time = datetime.datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO traces (trace_id, dsl_file, start_time, status, input) VALUES (?, ?, ?, ?, ?)",
                (trace_id, dsl_file, start_time, "RUNNING", json.dumps(input_data, ensure_ascii=False))
            )
        return trace_id

    def end_trace(self, trace_id: str, status: str, output: Dict[str, Any]):
        """结束一次追踪。"""
        end_time = datetime.datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE traces SET end_time = ?, status = ?, output = ? WHERE trace_id = ?",
                (end_time, status, json.dumps(output, ensure_ascii=False), trace_id)
            )

    def log_event(self, trace_id: str, step_name: str, event_type: str, details: Any):
        """记录具体的执行事件。"""
        timestamp = datetime.datetime.now()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (trace_id, timestamp, step_name, event_type, details) VALUES (?, ?, ?, ?, ?)",
                (trace_id, timestamp, step_name, event_type, json.dumps(details, ensure_ascii=False))
            )

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """获取完整的追踪记录。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 获取 trace 元数据
            cursor.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
            trace = cursor.fetchone()
            if not trace:
                return None
            
            trace_dict = dict(trace)
            if trace_dict['input']: trace_dict['input'] = json.loads(trace_dict['input'])
            if trace_dict['output']: trace_dict['output'] = json.loads(trace_dict['output'])
            
            # 获取 events
            cursor.execute("SELECT * FROM events WHERE trace_id = ? ORDER BY id", (trace_id,))
            events = [dict(row) for row in cursor.fetchall()]
            for event in events:
                if event['details']:
                    try:
                        event['details'] = json.loads(event['details'])
                    except:
                        pass # 保持原样
            
            trace_dict["events"] = events
            return trace_dict
