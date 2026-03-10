const Database = require('better-sqlite3');
const path = require('path');

// 订单数据库路径
const ORDERS_DB_PATH = path.join(__dirname, 'orders.db');
// 检查点数据库路径
const CHECKPOINTS_DB_PATH = path.join(__dirname, 'checkpoints.db');

class OrderDatabase {
    constructor() {
        this.conn = new Database(ORDERS_DB_PATH);
        this.init();
    }

    init() {
        // 创建订单表
        this.conn.exec(`
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT,
                status TEXT,
                items TEXT,
                logistics_info TEXT,
                created_at TEXT
            )
        `);

        // 检查是否已有数据
        const count = this.conn.prepare('SELECT count(*) as count FROM orders').get().count;
        if (count === 0) {
            console.log('正在写入示例订单数据...');
            const insert = this.conn.prepare(
                'INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)'
            );
            const now = new Date().toISOString();
            insert.run('12345', 'user_001', 'shipped', 'Wireless Headphones', 'Arrived at Beijing Sorting Center', now);
            insert.run('67890', 'user_001', 'pending_payment', 'Smart Watch', 'Waiting for payment', now);
            insert.run('11223', 'user_002', 'delivered', 'Laptop Stand', 'Delivered to locker', now);
        }
    }

    checkOrder(orderId) {
        const stmt = this.conn.prepare(
            'SELECT status, items, logistics_info FROM orders WHERE order_id = ?'
        );
        const result = stmt.get(orderId);

        if (result) {
            return `订单 ${orderId}（${result.items}）：当前状态为『${result.status}』。物流信息：${result.logistics_info}。`;
        } else {
            return `未查询到订单 ${orderId}，请检查订单号是否正确。`;
        }
    }

    close() {
        this.conn.close();
    }
}

class CheckpointDatabase {
    constructor() {
        this.conn = new Database(CHECKPOINTS_DB_PATH);
        this.init();
    }

    init() {
        // 创建检查点表
        this.conn.exec(`
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT,
                checkpoint_id TEXT PRIMARY KEY,
                parent_checkpoint_id TEXT,
                checkpoint_data TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        `);
    }

    saveCheckpoint(threadId, checkpointId, parentCheckpointId, checkpointData, metadata) {
        const stmt = this.conn.prepare(`
            INSERT OR REPLACE INTO checkpoints
            (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata)
            VALUES (?, ?, ?, ?, ?)
        `);
        stmt.run(
            threadId,
            checkpointId,
            parentCheckpointId,
            JSON.stringify(checkpointData),
            JSON.stringify(metadata)
        );
    }

    getLatestCheckpoint(threadId) {
        const stmt = this.conn.prepare(`
            SELECT * FROM checkpoints
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        `);
        const result = stmt.get(threadId);

        if (result) {
            return {
                threadId: result.thread_id,
                checkpointId: result.checkpoint_id,
                parentCheckpointId: result.parent_checkpoint_id,
                checkpointData: JSON.parse(result.checkpoint_data),
                metadata: JSON.parse(result.metadata)
            };
        }
        return null;
    }

    close() {
        this.conn.close();
    }
}

module.exports = {
    OrderDatabase,
    CheckpointDatabase
};
