# 酒店评论分析智能体

基于 **LangChain + Ollama** + 向量检索（FAISS），实现的中文酒店评论情感与内容分析项目。  
支持用自然语言输入多样问题，自动分析大量真实酒店用户评论。

---

## 项目简介

本项目致力于帮助用户/运营者从大量酒店评论中，智能抽取、归纳和分析情感及高频话题。  
利用大语言模型结合向量检索，极大提升分析速度与准确性。适用于：

- 酒店运营分析（情感趋势、常见优缺点评估等）
- 舆情监控和服务改善
- 数据分析与教育演示

---

## 系统架构

- **前端**：简单网页界面，输入问题/查看分析结果（`frontend/index.html`）
- **后端**：FastAPI 服务，统一对外提供接口（`api/main.py`）
- **核心Agent**：评论向量化/检索/大模型分析（`agent/agent.py`）
- **数据存储**：FAISS 向量数据库，支持高效检索

---

## 快速开始

### 1. 环境准备

- Python 3.8+
- 推荐使用 [pipenv](https://pipenv.pypa.io/) 或 `python -m venv venv`
- [Ollama](https://ollama.com/)（本地大模型服务，支持 Qwen2/Baichuan/Llama3 等）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型 & Ollama

- 启动 Ollama 服务（首次运行安装 [Ollama](https://ollama.com/download)）
- 下载模型（示例默认使用 阿里 Qwen2 1.5B，亦可换成 Llama3/Baichuan/Mistral等）

```bash
ollama serve
ollama pull qwen2:1.5b
```

> 如需其他模型，参考 [Ollama 文档](https://ollama.com/library)

### 4. 准备数据

- 默认有示例数据：`data/hotel_comments.csv`  
  只需列名有 `review` 字段即可。

- 如需扩充，添加/替换 `data/hotel_comments.csv` 即可。

### 5. 启动服务

```bash
cd api
python main.py
```

- 启动后会自动构建/加载向量数据库，后端和前端都自动就绪。
- 浏览器访问：http://localhost:8000

---

## 使用示例

1. 打开网页输入你的分析问题。例如：
   - `分析所有服务方面的好评和差评`
   - `统计好评有多少条`
   - `找出关于房间的好评`
   - `分析早餐相关的评论`
2. 稍等片刻，系统直接展示智能分析报告。

---

## 常见问题与说明

- **Ollama 连接失败**  
  请确保本地 Ollama 已运行、模型已下载，且监听11434端口（默认）。
- **中文支持**  
  推荐 Qwen2、Baichuan2、Llama3-8B等主流大模型，中文理解较好。
- **评论数量**  
  数据量较大初次向量化需等待几分钟，以后会高速加载。

---

## 主要依赖

- [LangChain](https://github.com/langchain-ai/langchain)
- [langchain-ollama](https://github.com/langchain-ai/langchain-ollama)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.com/)

---

## 项目结构

```
project-root/
│
├── agent/
│   └── agent.py         # 评论分析智能体核心
├── api/
│   └── main.py          # FastAPI 后端
├── frontend/
│   └── index.html       # 简易前端页面
├── data/
│   └── hotel_comments.csv  # 样本评论数据
├── requirements.txt
└── README.md
```

---

## 致谢

- 感谢开源社区和大模型/工具开发者！

---

如遇问题或建议，欢迎 [issues 或 pr] 提交反馈。

