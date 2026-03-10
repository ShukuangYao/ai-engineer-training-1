const { OrderDatabase } = require('./database');
const { RAGRetriever } = require('./rag');

class Tools {
    constructor() {
        this.orderDB = new OrderDatabase();
        this.ragRetriever = new RAGRetriever();
    }

    checkOrder(orderId) {
        return this.orderDB.checkOrder(orderId);
    }

    searchPolicy(query) {
        return this.ragRetriever.search(query);
    }

    processAudioInput(filePath) {
        console.log(`[系统] 正在处理音频文件：${filePath}`);
        return "查订单 12345";
    }

    processImageInput(filePath) {
        console.log(`[系统] 正在处理图片文件：${filePath}`);
        return "图片中订单号似乎是 67890";
    }
}

module.exports = {
    Tools
};
