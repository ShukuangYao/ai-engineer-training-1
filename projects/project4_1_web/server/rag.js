class RAGRetriever {
    constructor() {
        this.policies = [
            "退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。",
            "物流政策：满 50 美元免邮，标准配送一般为 3-5 个工作日。",
            "质保政策：电子类商品享受 1 年制造商质保服务。",
            "支付政策：支持信用卡、PayPal 和支付宝。",
            "订单修改：订单状态变为\"已发货\"后不可再修改订单信息。"
        ];
        this.documents = this.policies.map((p, i) => ({
            id: i,
            content: p,
            metadata: { source: "policy_doc" }
        }));
    }

    // 简单的关键词匹配检索（演示用）
    searchPolicy(query) {
        const keywords = query.toLowerCase().split(/\s+/);
        const results = this.documents.filter(doc => {
            const content = doc.content.toLowerCase();
            return keywords.some(keyword => content.includes(keyword));
        });

        // 如果没有找到匹配，返回前两条
        if (results.length === 0) {
            return this.documents.slice(0, 2).map(d => d.content);
        }

        return results.map(d => d.content);
    }

    search(query) {
        const results = this.searchPolicy(query);
        return results.join('\n');
    }
}

module.exports = {
    RAGRetriever
};
