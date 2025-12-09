# agent/rag_agent.py
import os
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

class HotelRAGAgent:
    def __init__(self, data_path="data/hotel_comments.csv"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"评论文件 {data_path} 不存在！")
        
        # 读取 CSV 文件并提取 review 列
        df = pd.read_csv(data_path, encoding="utf-8")
        if "review" not in df.columns:
            raise ValueError(f"CSV 文件中未找到 'review' 列。可用列：{df.columns.tolist()}")
        
        # 提取 review 列，过滤空值
        comments = df["review"].dropna().astype(str).tolist()
        comments = [comment.strip() for comment in comments if comment.strip()]

        # 1. 加载 Embedding 模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. 构建或加载 FAISS 向量库
        vector_db_path = "data/faiss_vector_db"
        
        # 如果向量数据库已存在，直接加载
        if os.path.exists(vector_db_path):
            print("加载已存在的向量数据库...")
            self.vector_db = FAISS.load_local(
                vector_db_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"向量数据库加载完成，包含 {self.vector_db.index.ntotal} 条向量")
        else:
            # 如果不存在，创建新的向量数据库并保存
            print("正在构建向量数据库（首次运行需要一些时间）...")
            self.vector_db = FAISS.from_texts(comments, embeddings)
            # 保存向量数据库到本地
            os.makedirs(vector_db_path, exist_ok=True)
            self.vector_db.save_local(vector_db_path)
            print(f"向量数据库构建完成并已保存，包含 {len(comments)} 条向量")

        # 3. 加载 Ollama 大模型
        print("正在初始化 Ollama 模型连接...")
        self.llm = OllamaLLM(model="qwen2:1.5b", temperature=0.1)

        # 4. 构造 Prompt 模板
        prompt_template = """你是一个酒店评论分析助手，你能精准分析每条评论的情感。请根据以下真实用户评论回答问题。

相关评论：
{context}

问题：{question}

请总结回答，不要编造信息，输出相关评论，不要解释过程。
"""
#         prompt_template = """你是一个酒店运营分析师。请根据以下用户评论和问题，生成一份结构化分析报告，包含：
# 1. 总体情感倾向（正面/负面比例）
# 2. 高频正面关键词
# 3. 高频负面关键词
# 4. 2条典型差评摘录

# 评论列表：
# {context}
# 问题：{question}
# 请根据用户的问题输出分析，不要任何解释：

# """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 5. 创建 RAG 链
        # 为 map_reduce 链创建不同的 prompt
        question_prompt = PromptTemplate(
            template="""基于以下评论内容，回答问题。
请用三句话句话总结回答，不要编造信息，输出相关评论，不要解释过程。

评论：{context}
问题：{question}

回答：""",
            input_variables=["context", "question"]
        )

        combine_prompt = PromptTemplate(
            template="""基于以下各个评论的回答，综合生成最终答案。
请用三句话句话总结回答，不要编造信息，输出相关评论，不要解释过程。

各部分的回答：
{summaries}

最终问题：{question}

最终回答：""",
            input_variables=["summaries", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_reduce",
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 20}),
            return_source_documents=False,
            chain_type_kwargs={
                "question_prompt": question_prompt,
                "combine_prompt": combine_prompt
            }
        )

    def ask(self, query: str) -> str:
        try:
            result = self.qa_chain.invoke({"query": query})
            return result["result"].strip()
        except Exception as e:
            error_msg = str(e)
            # 检查是否是 Ollama 连接错误
            if "Connection" in error_msg or "连接" in error_msg or "11434" in error_msg or "Max retries exceeded" in error_msg:
                return (
                    "无法连接到 Ollama 服务。\n"
                    "解决方案：\n"
                    "1. 确保 Ollama 服务已启动（在终端运行：ollama serve）\n"
                    "2. 确保模型已下载（运行：ollama pull qwen2:1.5b）\n"
                    "3. 检查 Ollama 是否运行在默认端口 11434"
                )
            return f"Agent 处理失败: {error_msg}"
if __name__ == "__main__":
    h = HotelRAGAgent()
    result = h.ask("你是一个酒店评论情感分析师，分析所有服务方面是好评和差评")
    print(result)